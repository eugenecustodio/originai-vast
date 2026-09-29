"""Video frame sources for the detector: GPU (NVDEC) decoding first, CPU as fallback.

All sources yield RGB uint8 tensors shaped (B, 3, H, W) on the compute device, turned
upright for rotated (phone) videos, together with each frame's presentation timestamp
and its index in presentation order (the index the thesis decoder recorded).

Order of preference:
  1. torchcodec on CUDA, "nvdec" backend: NVDEC hardware decoding through the driver's
     NVCUVID library; frames never leave the GPU.
  2. torchcodec on CUDA, "ffmpeg" backend (FFmpeg's CUDA hwaccel).
  3. torchcodec on CPU: FFmpeg decoding, bit-identical to the thesis PyAV rgb24 frames.
  4. PyAV: the thesis decoder itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import torch

log = logging.getLogger("originai.video")


@dataclass
class VideoInfo:
    duration: float
    fps: float | None
    width: int  # upright
    height: int
    rotation: int  # degrees applied to make the frame upright
    num_frames: int | None


def probe(path: Path) -> tuple[float, float | None, int, int, int, int | None]:
    """Duration (same rule as the thesis inventory), fps, raw size, rotation, frame count."""
    with av.open(str(path), mode="r", metadata_errors="ignore") as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError("decode_failed")
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        elif stream.frames and stream.average_rate:
            duration = float(stream.frames / stream.average_rate)
        else:
            raise ValueError("metadata_error")
        fps = float(stream.average_rate) if stream.average_rate else None
        rotation = 0
        for frame in container.decode(stream):
            rotation = int(round(getattr(frame, "rotation", 0) or 0))
            width, height = frame.width, frame.height
            break
        else:
            raise ValueError("decode_failed")
        return duration, fps, width, height, rotation, (int(stream.frames) or None)


class TorchcodecSource:
    """Decode with torchcodec (NVDEC when ``device`` is CUDA)."""

    def __init__(self, path: Path, device: torch.device, raw_size: tuple[int, int], rotation: int,
                 cuda_backend: str | None = None) -> None:
        import contextlib

        from torchcodec.decoders import VideoDecoder, set_cuda_backend

        self.device = device
        kwargs = {"device": str(device), "seek_mode": "exact", "dimension_order": "NCHW"}
        if device.type == "cpu":
            kwargs["num_ffmpeg_threads"] = 0  # let FFmpeg use every core
        backend = set_cuda_backend(cuda_backend) if cuda_backend else contextlib.nullcontext()
        with backend:
            self.decoder = VideoDecoder(str(path), **kwargs)
        self.num_frames = int(self.decoder.metadata.num_frames)
        self.name = f"torchcodec-{device.type}" + (f"-{cuda_backend}" if cuda_backend else "")
        self._raw_w, self._raw_h = raw_size
        self._turn = (rotation // 90) % 4
        self._fix_rotation: bool | None = None

    def batches(self, batch: int) -> Iterator[tuple[torch.Tensor, np.ndarray, np.ndarray]]:
        for start in range(0, self.num_frames, batch):
            stop = min(start + batch, self.num_frames)
            frames = self.decoder.get_frames_in_range(start, stop)
            data = frames.data
            if data.device != self.device:
                data = data.to(self.device, non_blocking=True)
            if self._fix_rotation is None:  # does this build turn frames upright itself?
                h, w = data.shape[2:]
                self._fix_rotation = self._turn in (1, 3) and (w, h) == (self._raw_w, self._raw_h)
                fallback = getattr(self.decoder, "cpu_fallback", None)
                if fallback:  # NVDEC could not be used; torchcodec decoded on the CPU instead
                    self.name = f"{self.name} -> CPU ({str(fallback).split('due to: ')[-1]})"
            if self._fix_rotation:
                data = torch.rot90(data, self._turn, dims=(2, 3))
            yield data, frames.pts_seconds.double().cpu().numpy(), np.arange(start, stop)


class PyAVSource:
    """The thesis decoder: PyAV, rgb24, first frame at or after each timestamp."""

    name = "pyav-cpu"

    def __init__(self, path: Path, device: torch.device) -> None:
        self.path, self.device = path, device
        self.num_frames = None

    def batches(self, batch: int) -> Iterator[tuple[torch.Tensor, np.ndarray, np.ndarray]]:
        buf, pts, idx = [], [], []
        with av.open(str(self.path), mode="r", metadata_errors="ignore") as container:
            stream = next(s for s in container.streams if s.type == "video")
            stream.thread_type = "AUTO"
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            for i, frame in enumerate(container.decode(stream)):
                if frame.pts is not None and frame.time_base is not None:
                    t = float(frame.pts * frame.time_base)
                elif fps:
                    t = i / fps
                else:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                turn = int(round((getattr(frame, "rotation", 0) or 0) / 90)) % 4
                if turn:
                    rgb = np.rot90(rgb, turn)
                buf.append(torch.from_numpy(np.ascontiguousarray(rgb)))
                pts.append(t)
                idx.append(i)
                if len(buf) == batch:
                    yield torch.stack(buf).permute(0, 3, 1, 2).to(self.device), np.array(pts), np.array(idx)
                    buf, pts, idx = [], [], []
        if buf:
            yield torch.stack(buf).permute(0, 3, 1, 2).to(self.device), np.array(pts), np.array(idx)


def nvdec_diagnostics() -> dict:
    """Why NVDEC may be unavailable: driver capabilities and the NVCUVID library."""
    import ctypes.util
    import os

    return {
        "NVIDIA_DRIVER_CAPABILITIES": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
        "libnvcuvid": ctypes.util.find_library("nvcuvid"),
    }


def open_source(path: Path, device: torch.device, prefer_gpu: bool = True):
    """Return (source, VideoInfo). Falls back from NVDEC to CPU decoding if needed."""
    duration, fps, raw_w, raw_h, rotation, frames = probe(path)
    upright = (raw_h, raw_w) if (rotation // 90) % 2 else (raw_w, raw_h)
    info = VideoInfo(duration, fps, upright[0], upright[1], rotation % 360, frames)
    attempts: list[tuple[torch.device, str | None]] = []
    if prefer_gpu and device.type == "cuda":
        attempts += [(device, "nvdec"), (device, "ffmpeg")]
    attempts.append((torch.device("cpu"), None))
    for decode_device, backend in attempts:
        try:
            source = TorchcodecSource(path, decode_device, (raw_w, raw_h), rotation, backend)
            if decode_device.type == "cuda":
                next(iter(source.batches(1)))  # fail here, not mid-analysis, if NVDEC is unusable
            if decode_device.type == "cpu" and device.type == "cuda":
                source.device = device  # decode on CPU, compute on GPU
            info.num_frames = source.num_frames
            return source, info
        except Exception as exc:  # no NVDEC / codec unsupported / torchcodec missing
            log.warning("torchcodec %s/%s unavailable (%s: %s)", decode_device, backend, type(exc).__name__, str(exc)[:160])
    return PyAVSource(path, device), info
