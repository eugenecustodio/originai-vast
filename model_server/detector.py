"""The deepfake detectors: face pipeline on the GPU, every-frame scoring, evidence maps.

Two kinds of model are served (``arch`` in models.json): the thesis's Video Swin-B
detectors, described below, and DFD-FCG (dfd_fcg.py), which reads its own face crop and
10-frame clips but shares the decoding, face detection, tracking, timeline and rendering.

Preprocessing reproduces the training repository
(Thesis_Code/src/swinfusionpp/preprocessing.py, policy ``preprocessing_v001``) with the
heavy steps moved to the GPU and verified bit-exact against the originals:

    decode (NVDEC via torchcodec; CPU fallback identical to the thesis PyAV rgb24)
    -> MTCNN (gpu_mtcnn: batched facenet-pytorch detector, identical detections),
       keep detections with probability >= 0.90
    -> five-landmark similarity alignment to the ArcFace template, 224x224 RGB uint8
       (fixed-point warp identical to OpenCV 4.11 warpAffine, INTER_LINEAR, REFLECT_101)
    -> FaceNet (vggface2) identity tracking of the primary face.

Every decoded frame with the tracked face is analysed. The model only takes 16 frames at
the spacing it was trained on (about 0.25-0.35 s apart), so the video is split into
interleaved phases: phase p holds frames p, p+k, p+2k, ... where k = round(spacing x fps).
Each phase is tiled into 16-frame clips, so every face frame is scored once, always at
the trained frame spacing. The verdict follows the thesis read-out over all clips:
mean clip logit -> sigmoid(logit / T) -> threshold.

The exact thesis evaluation (64 timestamps -> 48 tracked frames -> three anchors) is
reported alongside as ``protocol``.

Evidence maps: the Video Swin-B head is LayerNorm -> mean pool -> linear, so each
final-stage token (8 x 7 x 7 per clip) contributes exactly ``w . LayerNorm(token)`` to the
clip logit; the mean of those contributions plus the bias *is* the logit. This class
activation map (Grad-CAM at the final norm layer) comes free with the scoring pass and is
rendered on the GPU, relative within the video.
"""

from __future__ import annotations

import base64
import contextlib
import gc
import json
import logging
import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.io import encode_jpeg
from torchvision.models.video import swin3d_b

import dfd_fcg
from gpu_mtcnn import detect_faces
from video_io import nvdec_diagnostics, open_source

log = logging.getLogger("originai.detector")

FAILURE_MESSAGES = {
    "metadata_error": "The video's duration could not be read.",
    "decode_failed": "The video could not be decoded.",
    "no_face": "No face was detected in the video.",
    "low_detection_confidence": "A face was found, but not clearly enough to analyze.",
    "identity_track_failed": "A consistent face could not be followed through the video.",
    "multiple_face_ambiguity": "Several similar faces made it impossible to follow one person.",
    "insufficient_valid_frames": "A clear face is visible in too few frames to analyze.",
    "insufficient_valid_clips": "A clear face is visible in too few frames to analyze.",
}


class AnalysisFailed(Exception):
    """Preprocessing could not produce enough face evidence for a result."""

    def __init__(self, reason: str) -> None:
        super().__init__(FAILURE_MESSAGES.get(reason, reason))
        self.reason = reason


def _is_oom(exc: BaseException) -> bool:
    """CUDA out of memory, including the cuBLAS/cuDNN allocation failures raised as RuntimeError."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and ("out of memory" in str(exc) or "ALLOC_FAILED" in str(exc))


# ---------------------------------------------------------------------------
# Running several scans on one GPU
# ---------------------------------------------------------------------------
# One scan keeps the GPU busy only part of the time: much of it is decoding, face tracking and
# waiting on small GPU steps. Scans therefore run side by side in threads (ScanGate), so one
# scan's GPU work fills the gaps in another's. Measured on an RTX 5090 with 12 CPU cores (mixed
# workload, scans/min): 1 at a time 4.4, 2 at once 7.4, 3 at once 8.7, 4 at once 8.0 (the CPU, which
# decodes the videos, was then the limit). With 3 at once and the heaviest videos the GPU peaked
# at 24 GB: the models take ~3.7 GB, and each scan about 6.7 GB including its share of the model pass.
GPU_RESERVE_GB = 6.0  # models, CUDA context and allocator slack
GPU_GB_PER_SCAN = 7.5
CPU_CORES_PER_SCAN = 3.5  # decoding and the Python side of face detection and tracking
HOST_RESERVE_GB = 6.0
HOST_GB_PER_SCAN = 3.5  # frame previews of a 2-minute 60 fps video, decoder buffers
MAX_SLOTS = 4


def _host_available_gb() -> float:
    """Memory this container can still use (MemAvailable, capped by its cgroup limit)."""
    try:
        with open("/proc/meminfo") as f:
            available = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable")) / 2**20
    except (OSError, StopIteration, ValueError):
        return float("inf")
    for limit_file, usage_file in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                   ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            with open(limit_file) as f:
                limit = f.read().strip()
            if limit != "max" and int(limit) < 1 << 50:
                with open(usage_file) as f:
                    available = min(available, (int(limit) - int(f.read())) / 2**30)
            break
        except (OSError, ValueError):
            continue
    return available


def _cpu_cores() -> float:
    """CPU cores this container may use (its cgroup quota, else the cores it may run on)."""
    for quota_file, period_file in (("/sys/fs/cgroup/cpu.max", None),
                                    ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            with open(quota_file) as f:
                fields = f.read().split()
            if period_file is None:
                if fields[0] != "max":
                    return int(fields[0]) / int(fields[1])
            elif int(fields[0]) > 0:
                with open(period_file) as f:
                    return int(fields[0]) / int(f.read())
        except (OSError, ValueError, IndexError):
            continue
    return float(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))


def scan_slots(device: torch.device) -> int:
    """How many scans run at once: ORIGINAI_CONCURRENCY, or what GPU memory, CPU cores and host
    memory allow (3 on an RTX 5090 with 12 or more cores, 2 on a 24 GB card, 1 on a 16 GB card)."""
    configured = int(os.environ.get("ORIGINAI_CONCURRENCY", "0") or 0)
    if configured > 0:
        return configured
    if device.type != "cuda":
        return 1
    gpu_gb = torch.cuda.get_device_properties(device).total_memory / 2**30
    by_gpu = int((gpu_gb - GPU_RESERVE_GB) // GPU_GB_PER_SCAN)
    by_cpu = int(_cpu_cores() // CPU_CORES_PER_SCAN)
    by_host = int((_host_available_gb() - HOST_RESERVE_GB) // HOST_GB_PER_SCAN)
    return max(1, min(MAX_SLOTS, by_gpu, by_cpu, by_host))


class ScanGate:
    """Admits up to ``slots`` scans at once, first come first served, each on its own CUDA stream.

    Per-scan streams keep scans out of each other's waits. Everything a scan puts on the GPU must
    then be ordered against its stream, which is why videos are decoded on the CPU and uploaded on
    that stream (``gpu_decode`` false in models.json): torchcodec's GPU decoder, and the CPU
    fallback inside it, order their uploads only against the default stream, and with per-scan
    streams that made results vary and, with several scans, broke the GPU context. The JPEG
    encoder gets the same treatment in ``_jpegs``; the model pass runs on a shared stream of its
    own (``DetectorService._model_pass``).

    ``exclusive`` admits a scan on its own: it goes to the front of the queue and waits for the
    running scans to finish. A scan that ran out of GPU memory next to others is re-run this way,
    so running scans side by side never makes a scan fail."""

    def __init__(self, slots: int, device: torch.device) -> None:
        self.slots = max(1, int(slots))
        self.device = device
        self._cond = threading.Condition()
        self._queue: deque = deque()
        self._running = 0
        self._exclusive = False
        # High priority: a scan's many small steps (face detection, alignment) start as soon as SMs
        # free up instead of queueing behind another scan's long model-pass kernels.
        self._streams = ([torch.cuda.Stream(device, priority=-1) for _ in range(self.slots)]
                         if device.type == "cuda" else [])

    def status(self) -> dict:
        with self._cond:
            return {"slots": self.slots, "running": self._running, "waiting": len(self._queue)}

    @contextlib.contextmanager
    def slot(self, exclusive: bool = False):
        ticket = object()
        with self._cond:
            (self._queue.appendleft if exclusive else self._queue.append)(ticket)

            def admitted() -> bool:
                if self._queue[0] is not ticket:
                    return False
                return self._running == 0 if exclusive else self._running < self.slots and not self._exclusive

            self._cond.wait_for(admitted)
            self._queue.popleft()
            self._running += 1
            self._exclusive = self._exclusive or exclusive
            stream = self._streams.pop() if self._streams else None
            self._cond.notify_all()
        try:
            with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                yield
        finally:
            if stream is not None:
                stream.synchronize()
            with self._cond:
                if stream is not None:
                    self._streams.append(stream)
                self._running -= 1
                if exclusive:
                    self._exclusive = False
                idle = self._running == 0
                self._cond.notify_all()
            if idle and self.device.type == "cuda":
                torch.cuda.empty_cache()  # give memory back only when no scan is running


class _LandmarkQueue:
    """Collects 2D-FAN inputs over frame batches and runs the network on full chunks only.

    The network always takes ``CHUNK`` (16) faces; a 1080p frame batch holds 19 frames, so calling
    it per batch padded 3 faces up to 16 every time and wasted ~40% of the work. Each face goes
    through the same 16-face call either way, so its landmarks are the same."""

    def __init__(self, landmarker: dfd_fcg.Landmarker) -> None:
        self.landmarker = landmarker
        self.resolved_upto = -1  # every face in frames up to this index has its landmarks (or none)
        self._inputs: list[torch.Tensor] = []  # waiting network inputs, in face order
        self._waiting = 0
        self._peaks: list[torch.Tensor] = []  # computed peaks not yet handed out
        self._ready = 0
        self._batches: deque = deque()  # (geometry, candidates, kept faces, last frame) per frame batch, in order

    def add(self, inputs: torch.Tensor, geometry: dict, candidates: list["Candidate"], last_frame: int) -> None:
        """One frame batch's faces (possibly none), ``last_frame`` being the batch's last frame index."""
        self._batches.append((geometry, candidates, len(inputs), last_frame))
        if len(inputs):
            self._inputs.append(inputs)
            self._waiting += len(inputs)
        self._run(self._waiting - self._waiting % self.landmarker.CHUNK)

    def finish(self) -> None:
        self._run(self._waiting)

    @torch.inference_mode()
    def _run(self, take: int) -> None:
        if take:
            waiting = torch.cat(self._inputs) if len(self._inputs) > 1 else self._inputs[0]
            self._peaks.append(self.landmarker.run(waiting[:take]))
            rest = waiting[take:]
            self._inputs, self._waiting = ([rest] if len(rest) else []), len(rest)
            self._ready += take
        while self._batches and self._batches[0][2] <= self._ready:
            geometry, candidates, kept, last_frame = self._batches.popleft()
            peaks = torch.cat(self._peaks) if len(self._peaks) > 1 else (self._peaks[0] if self._peaks else None)
            points = self.landmarker.points(peaks[:kept] if kept else np.zeros((0, 68, 2)), geometry)
            self._peaks = [peaks[kept:]] if peaks is not None and len(peaks) > kept else []
            self._ready -= kept
            for candidate, face in zip(candidates, points):
                if np.isfinite(face).all():
                    object.__setattr__(candidate, "points68", face)  # set once, before anything reads it
            self.resolved_upto = last_frame


class _CanvasCrops:
    """DFD-FCG's face crops, cut during the one decoding pass.

    Each tracked face is cut from its full-resolution frame with a transform that needs the
    landmarks of the tracked faces up to 6 frames later, which used to take a second pass over
    the video. The tracker only looks back, so its choice for a frame is final once the frame is
    read; a crop is cut as soon as the frames 6 ahead are tracked and their landmarks are in,
    from the few recent frame batches kept on the GPU. Same tracker, same transforms, same pixels
    (decoding is deterministic): the crops equal those of the second pass."""

    def __init__(self, policy: dict[str, Any], landmarks: _LandmarkQueue) -> None:
        self.tracker = IdentityTracker(policy)
        self.landmarks = landmarks
        self.last_frame = -1
        self.kept: list[tuple[Candidate, tuple[np.ndarray, int, int]]] = []  # in frame order
        self.crops: torch.Tensor | None = None  # (len(kept), 150, 150, 3) uint8 once finished
        self._parts: list[torch.Tensor] = []
        self._unresolved: deque = deque()  # tracked faces whose landmarks are not in yet
        self._frames = np.empty(256, dtype=np.int64)  # tracked faces with landmarks: frame index ...
        self._points = np.empty((256, 68, 2), dtype=np.float32)  # ... and landmarks, in frame order
        self._faces: list[Candidate] = []
        self._next = 0  # first of those without a crop transform yet
        self._pixels: dict[int, tuple[torch.Tensor, int, int, int]] = {}  # frame index -> (batch, position, h, w)

    def add_batch(self, records: list[FrameRecord], flat: torch.Tensor | None, height: int, width: int) -> None:
        """One frame batch after detection; ``flat`` is the batch as (B, H*W, 3) (None if it has no faces)."""
        for position, record in enumerate(records):
            self.last_frame = max(self.last_frame, record.frame_index)
            if record.dense_pos < 0:
                continue  # protocol-only frames are not tracked
            selected = self.tracker.step(record.candidates)
            if selected is not None:
                self._unresolved.append(selected)
                self._pixels[record.frame_index] = (flat, position, height, width)
        self._advance(None)

    @torch.inference_mode()
    def finish(self, total_frames: int) -> None:
        self._advance(total_frames)
        self._pixels.clear()
        self.crops = (torch.cat(self._parts) if self._parts else None)
        self._parts = []

    @torch.inference_mode()
    def _advance(self, total_frames: int | None) -> None:
        final = total_frames is not None
        resolved = math.inf if final else self.landmarks.resolved_upto
        while self._unresolved and self._unresolved[0].record.frame_index <= resolved:
            face = self._unresolved.popleft()
            if face.points68 is not None:
                n = len(self._faces)
                if n == len(self._frames):
                    self._frames = np.concatenate([self._frames, np.empty_like(self._frames)])
                    self._points = np.concatenate([self._points, np.empty_like(self._points)])
                self._frames[n], self._points[n] = face.record.frame_index, face.points68
                self._faces.append(face)
        # Every face up to ``horizon`` is tracked and has its landmarks; a transform reads 6 frames ahead.
        horizon = math.inf if final else min(self.last_frame, self.landmarks.resolved_upto)
        total = total_frames if final else self.last_frame + 1  # only frames >= 6 from the end are done early
        n = len(self._faces)
        ready = []
        while self._next < n and self._frames[self._next] + dfd_fcg.SMOOTH_WINDOW // 2 <= horizon:
            geometry = dfd_fcg.frame_crop_geometry(self._points[:n], self._frames[:n], self._next, total)
            if geometry is not None:
                ready.append((self._faces[self._next], geometry))
            self._next += 1
        groups: dict[int, list] = {}
        for face, geometry in ready:
            groups.setdefault(id(self._pixels[face.record.frame_index][0]), []).append((face, geometry))
        for members in groups.values():
            flat, _, height, width = self._pixels[members[0][0].record.frame_index]
            positions = torch.tensor([self._pixels[f.record.frame_index][1] for f, _ in members], device=flat.device)
            self._parts.append(warp_faces(flat, height, width, positions, np.stack([g[0] for _, g in members]),
                                          dfd_fcg.CROP_SIZE, offsets=np.array([g[1:] for _, g in members]), zero_border=True))
            self.kept.extend(members)
        # Keep only the frames a later crop can still need.
        oldest = min(self._unresolved[0].record.frame_index if self._unresolved else math.inf,
                     self._frames[self._next] if self._next < n else math.inf)
        for frame in [f for f in self._pixels if f < oldest]:
            del self._pixels[frame]


class _Prefetcher:
    """Decodes batches in a background thread while the caller's GPU work runs.

    The thread queues its work (host-to-GPU copies, rotation) on the caller's CUDA stream, so the
    caller's later steps on that stream see finished frames."""

    def __init__(self, source, batch_size: int, stream: torch.cuda.Stream | None, depth: int = 3) -> None:
        self.wait = 0.0
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()

        def produce() -> None:
            try:
                with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                    for item in source.batches(batch_size):
                        if self._stop.is_set():
                            return
                        self._queue.put(item)
            except Exception as exc:  # surfaced in the consumer
                self._queue.put(exc)
            finally:
                self._queue.put(None)

        self._thread = threading.Thread(target=produce, name="decode", daemon=True)
        self._thread.start()

    def __iter__(self):
        while True:
            started = time.perf_counter()
            item = self._queue.get()
            self.wait += time.perf_counter() - started
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self) -> None:
        """Stop the thread and drain the queue so it never blocks holding frames."""
        self._stop.set()
        while self._thread.is_alive():
            try:
                self._queue.get(timeout=0.2)
            except queue.Empty:
                pass


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Normalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 1, 3, 1, 1))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if not clip.is_floating_point():
            clip = clip.float().div_(255.0)
        return (clip - self.mean) / self.std


class VideoSwinDetector(nn.Module):
    """Input: B x T x C x H x W RGB, uint8 or float in [0, 1]. Output: B clip logits."""

    def __init__(self) -> None:
        super().__init__()
        self.normalizer = Normalizer()
        self.backbone = swin3d_b(weights=None)
        self.backbone.head = nn.Identity()
        self.classifier = nn.Linear(1024, 1)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(clip).permute(0, 2, 1, 3, 4)  # -> B x C x T x H x W
        return self.classifier(self.backbone(x)).squeeze(-1)

    def stage_activation(self, clip: torch.Tensor, layer: int) -> torch.Tensor:
        """Output of ``backbone.features[layer]`` (channels-last B x T' x H' x W' x C)."""
        backbone = self.backbone
        x = self.normalizer(clip).permute(0, 2, 1, 3, 4)
        x = backbone.pos_drop(backbone.patch_embed(x))
        for block in backbone.features[: layer + 1]:
            x = block(x)
        return x

    def readout(self, final_activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Clip logits and per-token logit contributions from the final-stage output.

        norm -> mean pool -> linear equals the mean of per-token ``w . norm(token)``
        plus the bias, so ``logits`` is identical to ``forward`` and ``contributions``
        (B x T' x H' x W') is an exact decomposition of it.
        """
        tokens = self.backbone.norm(final_activation.float())
        contributions = tokens @ self.classifier.weight[0].float()
        logits = contributions.mean(dim=(1, 2, 3)) + self.classifier.bias[0].float()
        return logits, contributions


@dataclass
class ModelEntry:
    id: str
    name: str
    dataset: str
    temperature: float
    threshold: float
    frame_spacing_s: float
    net: nn.Module
    epoch: int | None = None
    metrics: dict | None = None
    arch: str = "video_swin_b"
    clip_length: int = 16

    def public_info(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "arch": self.arch,
            "dataset": self.dataset,
            "threshold": self.threshold,
            "frame_spacing_s": self.frame_spacing_s,
            "epoch": self.epoch,
            "validation_metrics": self.metrics,
        }


def load_model(path: Path, device: torch.device) -> tuple[VideoSwinDetector, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = VideoSwinDetector()
    net.load_state_dict(ckpt["model_state"], strict=True)
    net.requires_grad_(False)
    net.eval().to(device)
    return net, ckpt


# ---------------------------------------------------------------------------
# Thesis preprocessing (ported from swinfusionpp/preprocessing.py)
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class FrameRecord:
    """One decoded frame that is analysed (dense) and/or used by the thesis protocol."""

    timestamp: float
    frame_index: int
    dense_pos: int = -1  # position among densely analysed frames (-1: protocol only)
    detected: bool = False
    candidates: list["Candidate"] = field(default_factory=list)
    preview_index: int = -1


@dataclass(frozen=True)
class Candidate:
    frame_position: int
    box: tuple[float, float, float, float]
    probability: float
    landmarks: tuple[tuple[float, float], ...]
    crop_index: int  # row in the aligned-crop store (on the compute device)
    embedding: np.ndarray
    # Additions for the web service (unused by tracking):
    matrix: np.ndarray  # full-frame -> crop similarity transform
    record: FrameRecord
    points68: np.ndarray | None = None  # 2D-FAN landmarks, only read for DFD-FCG


@dataclass(frozen=True)
class TrackResult:
    candidates: list[Candidate]
    association_costs: list[float]
    ambiguous_frames: int
    gap_frames: int
    termination_reason: str


def uniform_timestamps(duration: float, count: int, edge_fraction: float) -> np.ndarray:
    if duration <= 0 or count <= 0:
        raise ValueError("duration and count must be positive")
    start = duration * edge_fraction
    stop = max(start, duration * (1.0 - edge_fraction))
    return np.linspace(start, stop, count, dtype=np.float64)


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != (5, 2) or target.shape != (5, 2):
        raise ValueError("Five two-dimensional landmarks are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, vh = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vh
    variance = np.sum(source_centered**2) / source.shape[0]
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate landmarks")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    matrix = np.empty((2, 3), dtype=np.float32)
    matrix[:, :2] = (scale * rotation).astype(np.float32)
    matrix[:, 2] = translation.astype(np.float32)
    return matrix


def canonical_landmarks(size: int = 224) -> np.ndarray:
    template = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    return template * (size / 112.0)


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return math.inf
    return float(1.0 - np.dot(left, right) / (left_norm * right_norm))


class IdentityTracker:
    """The thesis tracker, one frame at a time. It only looks back, so its choice for a frame is
    final as soon as that frame's candidates are known (DFD-FCG crops rely on this)."""

    def __init__(self, policy: dict[str, Any]) -> None:
        settings = policy["tracking"]
        self.maximum_distance = float(settings["maximum_cosine_distance"])
        self.minimum_iou = float(settings["minimum_iou_for_spatial_fallback"])
        self.ambiguity_margin = float(settings["ambiguity_margin"])
        self.ema_weight = float(settings["embedding_ema"])
        self.track: list[Candidate] = []
        self.association_costs: list[float] = []
        self.reference_embedding: np.ndarray | None = None
        self.previous_box: tuple[float, float, float, float] | None = None
        self.ambiguous = 0
        self.frames = 0

    def step(self, candidates: list[Candidate]) -> Candidate | None:
        """The next frame's candidates -> the tracked face in it, or None."""
        self.frames += 1
        if not candidates:
            return None
        if self.reference_embedding is None:
            selected = max(candidates, key=lambda item: item.probability)
            selected_cost = 0.0
        else:
            scored = []
            for candidate in candidates:
                distance = cosine_distance(self.reference_embedding, candidate.embedding)
                iou = box_iou(self.previous_box, candidate.box) if self.previous_box else 0.0
                if distance <= self.maximum_distance or iou >= self.minimum_iou:
                    scored.append((distance + 0.15 * (1.0 - iou), distance, candidate))
            scored.sort(key=lambda item: (item[0], -item[2].probability))
            if not scored:
                return None
            if len(scored) > 1 and scored[1][0] - scored[0][0] < self.ambiguity_margin:
                self.ambiguous += 1
                return None
            if scored[0][1] > self.maximum_distance and box_iou(self.previous_box, scored[0][2].box) < self.minimum_iou:
                return None
            selected = scored[0][2]
            selected_cost = float(scored[0][0])
        self.track.append(selected)
        self.association_costs.append(selected_cost)
        if self.reference_embedding is None:
            self.reference_embedding = selected.embedding.astype(np.float32, copy=True)
        else:
            self.reference_embedding = self.ema_weight * self.reference_embedding + (1.0 - self.ema_weight) * selected.embedding
            norm = np.linalg.norm(self.reference_embedding)
            if norm:
                self.reference_embedding /= norm
        self.previous_box = selected.box
        return selected

    def result(self) -> TrackResult:
        return TrackResult(
            candidates=self.track,
            association_costs=self.association_costs,
            ambiguous_frames=self.ambiguous,
            gap_frames=self.frames - len(self.track),
            termination_reason=("completed_candidate_sequence" if self.track else "no_primary_identity_initialized"),
        )


def track_candidates(candidates_by_frame: list[list[Candidate]], policy: dict[str, Any]) -> TrackResult:
    tracker = IdentityTracker(policy)
    for candidates in candidates_by_frame:
        tracker.step(candidates)
    return tracker.result()


def deterministic_selection_indices(length: int, maximum: int) -> list[int]:
    if length <= maximum:
        return list(range(length))
    indices = np.linspace(0, length - 1, maximum).round().astype(int).tolist()
    if len(set(indices)) != maximum:
        raise RuntimeError("Deterministic frame selection produced duplicate indices")
    return indices


def normalized_anchor_starts(clip_count: int) -> list[float]:
    if clip_count <= 0:
        raise ValueError("clip_count must be positive")
    return np.linspace(0.0, 1.0, clip_count, dtype=np.float64).tolist()


def map_normalized_anchors(frame_count: int, clip_length: int, normalized_starts: list[float]) -> list[list[int]]:
    if frame_count < clip_length or clip_length <= 0:
        return []
    maximum_start = frame_count - clip_length
    anchors = []
    for position in normalized_starts:
        if not 0.0 <= position <= 1.0:
            raise ValueError("Normalized anchor positions must lie in [0, 1]")
        start = round(position * maximum_start)
        anchors.append(list(range(start, start + clip_length)))
    return anchors


def evaluation_anchors(frame_count: int, clip_length: int, clip_count: int) -> list[list[int]]:
    if frame_count < clip_length * clip_count:
        return []
    return map_normalized_anchors(frame_count, clip_length, normalized_anchor_starts(clip_count))


def detection_is_accepted(probability: float | None, minimum_confidence: float) -> bool:
    return probability is not None and float(probability) >= minimum_confidence


# ---------------------------------------------------------------------------
# GPU face alignment: exact cv2.warpAffine(INTER_LINEAR, BORDER_REFLECT_101) reproduction
# ---------------------------------------------------------------------------
def _invert_affine(mats: np.ndarray) -> np.ndarray:
    """cv2.invertAffineTransform in float64, as warpAffine does internally."""
    m = mats.astype(np.float64)
    det = m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]
    inv_d = np.where(det != 0, 1.0 / np.where(det != 0, det, 1.0), 0.0)
    a11, a22 = m[:, 1, 1] * inv_d, m[:, 0, 0] * inv_d
    a12, a21 = -m[:, 0, 1] * inv_d, -m[:, 1, 0] * inv_d
    b1 = -a11 * m[:, 0, 2] - a12 * m[:, 1, 2]
    b2 = -a21 * m[:, 0, 2] - a22 * m[:, 1, 2]
    return np.stack([np.stack([a11, a12, b1], 1), np.stack([a21, a22, b2], 1)], 1)


def _reflect101(v: torch.Tensor, n: int) -> torch.Tensor:
    v = torch.where(v < 0, -v, v)
    v = torch.where(v >= n, 2 * (n - 1) - v, v)
    return v.clamp(0, n - 1)


def warp_faces(flat_frames: torch.Tensor, height: int, width: int, frame_inds: torch.Tensor,
               mats: np.ndarray, size: int, chunk: int = 32, offsets: np.ndarray | None = None,
               zero_border: bool = False) -> torch.Tensor:
    """Aligned crops (N, size, size, 3) uint8 from frames laid out as (B, H*W, 3).

    Fixed-point bilinear exactly as OpenCV computes it (AB_BITS=10, INTER_BITS=5,
    15-bit coefficients), verified pixel-identical to cv2.warpAffine 4.11.

    ``offsets`` (N, 2): each crop is the size x size window whose top-left corner is (x, y)
    of the warped image, i.e. cv2.warpAffine(...)[y : y + size, x : x + size]. The border is
    BORDER_REFLECT_101, or BORDER_CONSTANT 0 with ``zero_border``.
    """
    device = flat_frames.device
    inv = torch.from_numpy(_invert_affine(mats)).to(device)
    coords = torch.arange(size, dtype=torch.float64, device=device)
    origin = None if offsets is None else torch.as_tensor(np.asarray(offsets, dtype=np.float64), device=device)
    out = []
    for s in range(0, len(mats), chunk):
        m = inv[s : s + chunk]
        b = frame_inds[s : s + chunk].long()[:, None, None]
        xs = coords if origin is None else origin[s : s + chunk, 0:1] + coords
        ys = coords if origin is None else origin[s : s + chunk, 1:2] + coords
        adelta = torch.round(m[:, 0, 0, None] * xs * 1024).long()
        bdelta = torch.round(m[:, 1, 0, None] * xs * 1024).long()
        x0 = torch.round((m[:, 0, 1, None] * ys + m[:, 0, 2, None]) * 1024).long() + 16
        y0 = torch.round((m[:, 1, 1, None] * ys + m[:, 1, 2, None]) * 1024).long() + 16
        xx = (x0[:, :, None] + adelta[:, None, :]) >> 5  # rows = output y, cols = output x
        yy = (y0[:, :, None] + bdelta[:, None, :]) >> 5
        ix, fx, iy, fy = xx >> 5, xx & 31, yy >> 5, yy & 31
        if zero_border:
            c0, c1, r0, r1 = ix, ix + 1, iy, iy + 1

            def px(r, c):
                inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
                values = flat_frames[b, r.clamp(0, height - 1) * width + c.clamp(0, width - 1)].long()
                return values * inside[..., None]
        else:
            c0, c1 = _reflect101(ix, width), _reflect101(ix + 1, width)
            r0, r1 = _reflect101(iy, height), _reflect101(iy + 1, height)

            def px(r, c):
                return flat_frames[b, r * width + c].long()

        acc = (
            px(r0, c0) * ((32 - fx) * (32 - fy))[..., None]
            + px(r0, c1) * (fx * (32 - fy))[..., None]
            + px(r1, c0) * ((32 - fx) * fy)[..., None]
            + px(r1, c1) * (fx * fy)[..., None]
        )
        out.append(((acc * 32 + 16384) >> 15).clamp(0, 255).to(torch.uint8))
    return torch.cat(out) if out else torch.empty(0, size, size, 3, dtype=torch.uint8, device=device)


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------
# One-hue sequential ramp in the site's plum/orchid accent, applied to the relative
# evidence value (0 = median evidence in the shown clips or below, 1 = strongest):
# weaker regions stay clear, stronger evidence gets brighter and more opaque.
# The frontend legend mirrors these stops.
HEAT_STOPS = [
    (0.00, (116, 50, 130, 0)),
    (0.15, (130, 52, 150, 40)),
    (0.50, (170, 56, 190, 120)),
    (0.80, (205, 62, 218, 170)),
    (1.00, (230, 80, 236, 205)),
]


def _build_lut() -> np.ndarray:
    xs = np.linspace(0.0, 1.0, 256)
    stops = np.array([s[0] for s in HEAT_STOPS])
    colors = np.array([s[1] for s in HEAT_STOPS], dtype=np.float64)
    lut = np.stack([np.interp(xs, stops, colors[:, c]) for c in range(4)], axis=1)
    return lut.round().astype(np.uint8)


HEAT_LUT = _build_lut()


def _edge_feather(size: int, width: int = 18) -> np.ndarray:
    """Alpha multiplier that fades the overlay out toward the crop border."""
    ramp = np.clip(np.minimum(np.arange(size), np.arange(size)[::-1]) / width, 0.0, 1.0)
    return np.minimum.outer(ramp, ramp)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _jpegs(images: torch.Tensor, quality: int) -> list[bytes]:
    """(N, 3, H, W) uint8 -> JPEG bytes (nvJPEG on CUDA, libjpeg on CPU)."""
    images = [img.contiguous() for img in images]
    if not images or not images[0].is_cuda:
        return [e.numpy().tobytes() for e in encode_jpeg(images, quality=quality)]
    # torchvision's GPU encoder is tied to the stream that was current when it was first created
    # (the default stream, at warm-up): it waits only for that stream before reading the images,
    # and signals only that stream when the JPEG bytes are written. Scans run on their own
    # streams (ScanGate), so finish this scan's stream first and the default stream after.
    device = images[0].device
    torch.cuda.current_stream(device).synchronize()
    encoded = encode_jpeg(images, quality=quality)
    torch.cuda.default_stream(device).synchronize()
    return [e.cpu().numpy().tobytes() for e in encoded]


# ---------------------------------------------------------------------------
# Face pipeline
# ---------------------------------------------------------------------------
@dataclass
class DenseVideo:
    duration: float
    fps: float | None
    width: int
    height: int
    rotation: int
    decoder: str
    frames_decoded: int
    subsample: int
    records: list[FrameRecord]
    protocol_frames: list[FrameRecord]
    crops: torch.Tensor  # (N, 224, 224, 3) uint8 on the compute device
    previews: torch.Tensor  # (M, 3, ph, pw) uint8 in host memory (only the frames shown go back to the GPU)
    preview_size: tuple[int, int]
    preview_scale: float
    timings: dict[str, float]
    canvas: "_CanvasCrops | None" = None  # DFD-FCG: the tracked faces' crops, cut during the pass


class FacePipeline:
    def __init__(self, policy: dict[str, Any], device: torch.device, cfg: dict[str, Any]) -> None:
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.policy = policy
        self.device = device
        self.cfg = cfg
        detection = policy["face_detection"]
        mtcnn = MTCNN(
            keep_all=True,
            select_largest=False,
            min_face_size=int(detection["min_face_size"]),
            thresholds=list(map(float, detection["thresholds"])),
            factor=float(detection["factor"]),
            device=device,
            post_process=False,
        )
        self.pnet, self.rnet, self.onet = mtcnn.pnet, mtcnn.rnet, mtcnn.onet
        self.min_face = int(detection["min_face_size"])
        self.thresholds = list(map(float, detection["thresholds"]))
        self.factor = float(detection["factor"])
        self.min_conf = float(detection["minimum_confidence"])
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self.embedder.requires_grad_(False)
        self.size = int(policy["alignment"]["output_height"])
        self.template = canonical_landmarks(self.size)
        self.landmarker: dfd_fcg.Landmarker | None = None  # set by the service when DFD-FCG is served

    @torch.inference_mode()
    def _embed(self, crops: torch.Tensor) -> np.ndarray:
        """FaceNet embeddings: 160x160 resize + fixed_image_standardization, as in training."""
        out = []
        for s in range(0, len(crops), int(self.cfg["embed_batch"])):
            x = crops[s : s + int(self.cfg["embed_batch"])].permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False).round().clamp(0, 255)
            out.append(self.embedder((x - 127.5) / 128.0).float().cpu())
        return torch.cat(out).numpy() if out else np.empty((0, 512), np.float32)

    def _detection_batch(self, height: int, width: int) -> int:
        budget = int(self.cfg["detect_pixel_budget"])
        return int(min(32, max(2, budget // max(1, height * width))))

    def _sync(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()  # this scan's stream, not the whole GPU
        return time.perf_counter()

    def _stream(self) -> torch.cuda.Stream | None:
        return torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None

    @torch.inference_mode()
    def _process_batch(self, frames: torch.Tensor, records: list[FrameRecord], state: dict) -> None:
        """Detect, align, embed and keep previews for one batch of frames (all on the GPU)."""
        timing = state["timing"]
        t = self._sync()
        height, width = frames.shape[2:]
        results = detect_faces(frames, self.min_face, self.pnet, self.rnet, self.onet, self.thresholds, self.factor)
        t1 = self._sync()
        timing["detect"] += t1 - t
        inds, mats, meta = [], [], []
        for position, (record, (boxes, probs, points)) in enumerate(zip(records, results)):
            record.detected = len(boxes) > 0
            for box, prob, pts in zip(boxes, probs, points):
                if not detection_is_accepted(prob, self.min_conf):
                    continue
                try:
                    matrix = similarity_transform(np.asarray(pts), self.template)
                except ValueError:
                    continue
                inds.append(position)
                mats.append(matrix)
                meta.append((record, tuple(map(float, box)), float(prob), pts))
        landmark_time = 0.0
        flat, created = None, []
        if mats:
            flat = frames.permute(0, 2, 3, 1).reshape(len(frames), height * width, 3)
            crops = warp_faces(flat, height, width, torch.tensor(inds, device=frames.device), np.stack(mats), self.size)
            if state["canvas"] is None:
                flat = None  # only DFD-FCG cuts crops from the frames later
            embeddings = self._embed(crops)
            base = state["crop_count"]
            state["crops"].append(crops)
            state["crop_count"] += len(crops)
            for k, ((record, box, prob, pts), matrix, embedding) in enumerate(zip(meta, mats, embeddings)):
                candidate = Candidate(
                    frame_position=record.frame_index,
                    box=box,
                    probability=prob,
                    landmarks=tuple(tuple(map(float, p)) for p in pts),
                    crop_index=base + k,
                    embedding=embedding,
                    matrix=matrix,
                    record=record,
                )
                record.candidates.append(candidate)
                created.append(candidate)
        if state["fan"] is not None:  # DFD-FCG: 2D-FAN landmarks (filled in as full chunks run), then crops
            started = self._sync()
            boxes = dfd_fcg.sfd_boxes(np.array([m[1] for m in meta])) if meta else np.zeros((0, 4))
            inputs, geometry = self.landmarker.inputs(frames, inds, boxes)
            state["fan"].add(inputs, geometry, created, records[-1].frame_index)
            landmark_time = self._sync() - started
            timing["landmarks"] += landmark_time
            started = self._sync()
            state["canvas"].add_batch(records, flat, height, width)
            canvas_time = self._sync() - started
            timing["canvas"] += canvas_time
            landmark_time += canvas_time
        t2 = self._sync()
        timing["align_embed"] += t2 - t1 - landmark_time
        pw, ph = state["preview_size"]
        previews = frames.float() if (pw, ph) == (width, height) else F.interpolate(frames.float(), size=(ph, pw), mode="area")
        previews = previews.round().clamp(0, 255).to(torch.uint8)
        store, count = state["previews"], state["preview_count"]
        if count + len(previews) > len(store):  # header under-reported the frame count; rare
            grown = torch.empty((max(2 * len(store), count + len(previews)), *store.shape[1:]), dtype=store.dtype, device=store.device)
            grown[:count] = store[:count]
            state["previews"] = store = grown
        store[count : count + len(previews)].copy_(previews)  # preallocated host store: no per-batch allocations
        for record in records:
            record.preview_index = state["preview_count"]
            state["preview_count"] += 1
        timing["previews"] += self._sync() - t2

    def read(self, path: Path, spacing: float, landmarks: bool = False) -> DenseVideo:
        """``landmarks``: also find each face's 68 landmarks (DFD-FCG crops the face from them)."""
        cfg = self.cfg
        decoder_cfg = self.policy["decoder"]
        try:
            source, info = open_source(path, self.device, prefer_gpu=bool(cfg.get("gpu_decode", True)))
        except ValueError as exc:
            raise AnalysisFailed(str(exc) if str(exc) in FAILURE_MESSAGES else "decode_failed")
        except Exception as exc:
            if _is_oom(exc):
                raise
            raise AnalysisFailed("decode_failed")
        if info.duration <= 0:
            raise AnalysisFailed("metadata_error")

        # Thesis protocol targets: 64 uniform timestamps, first frame at or after each.
        targets = uniform_timestamps(info.duration, int(decoder_cfg["candidate_frame_count"]), float(decoder_cfg["edge_fraction"]))
        protocol: list[FrameRecord | None] = [None] * len(targets)
        pointer = 0
        # Dense set: every frame, or every m-th frame for very long / high-fps videos.
        expected = info.num_frames or int(info.duration * (info.fps or 30.0)) + 1
        subsample = max(1, math.ceil(expected / int(cfg["max_frames"])))

        pw = min(int(cfg["preview_width"]), info.width)
        preview_scale = pw / info.width
        ph = max(2, int(round(info.height * preview_scale)))
        capacity = expected // subsample + len(targets) + 8
        state = {
            "timing": {"decode_wait": 0.0, "detect": 0.0, "align_embed": 0.0, "previews": 0.0},
            "crops": [], "crop_count": 0, "preview_count": 0, "preview_size": (pw, ph),
            # Host memory: only the few frames shown with the evidence maps go back to the GPU,
            # so a long video does not hold up to ~3 GB of GPU memory next to other scans.
            "previews": torch.empty((capacity, 3, ph, pw), dtype=torch.uint8),
            "fan": _LandmarkQueue(self.landmarker) if landmarks and self.landmarker is not None else None,
        }
        state["canvas"] = _CanvasCrops(self.policy, state["fan"]) if state["fan"] is not None else None
        if state["fan"] is not None:
            state["timing"]["landmarks"] = 0.0
            state["timing"]["canvas"] = 0.0
        records: list[FrameRecord] = []
        pending: list[torch.Tensor] = []
        pending_records: list[FrameRecord] = []
        dense_count = 0
        decoded = 0
        batch_size = self._detection_batch(info.height, info.width)

        def flush() -> None:
            if pending:
                self._process_batch(torch.stack(pending), list(pending_records), state)
                pending.clear()
                pending_records.clear()

        # Decode the next batches in a background thread while the GPU works on this one.
        prefetch = _Prefetcher(source, batch_size, self._stream())
        try:
            for data, pts, idx in prefetch:
                for j in range(len(pts)):
                    decoded += 1
                    t, index = float(pts[j]), int(idx[j])
                    is_dense = index % subsample == 0
                    hits = []
                    while pointer < len(targets) and t + 1e-9 >= targets[pointer]:
                        hits.append(pointer)
                        pointer += 1
                    if not (is_dense or hits):
                        continue
                    record = FrameRecord(timestamp=t, frame_index=index)
                    if is_dense:
                        record.dense_pos = dense_count
                        dense_count += 1
                    for hit in hits:
                        protocol[hit] = record
                    frame = data[j]
                    if pending and pending[0].shape != frame.shape:
                        flush()  # resolution changed mid-stream
                    records.append(record)
                    pending.append(frame)
                    pending_records.append(record)
                    if len(pending) >= batch_size:
                        flush()
            flush()
        except AnalysisFailed:
            raise
        except Exception as exc:
            if _is_oom(exc):
                raise  # not a decoding problem: the scan is re-run (see DetectorService.analyze)
            if not records:
                raise AnalysisFailed("decode_failed")
            flush()
        finally:
            prefetch.close()
            state["timing"]["decode_wait"] = prefetch.wait
        if not records:
            raise AnalysisFailed("decode_failed")
        if state["fan"] is not None:  # the last, partial chunk of landmark inputs, then the last crops
            started = self._sync()
            state["fan"].finish()
            middle = self._sync()
            state["canvas"].finish(decoded)
            state["timing"]["landmarks"] += middle - started
            state["timing"]["canvas"] += self._sync() - middle

        crops = torch.cat(state["crops"]) if state["crops"] else torch.empty(0, self.size, self.size, 3, dtype=torch.uint8, device=self.device)
        previews = state["previews"][: state["preview_count"]]
        return DenseVideo(
            duration=info.duration,
            fps=info.fps,
            width=info.width,
            height=info.height,
            rotation=info.rotation,
            decoder=source.name,
            frames_decoded=decoded,
            subsample=subsample,
            records=records,
            protocol_frames=[r for r in protocol if r is not None],
            crops=crops,
            previews=previews,
            preview_size=(pw, ph),
            preview_scale=preview_scale,
            timings=dict(state["timing"]),
            canvas=state["canvas"],
        )

    def track(self, frames: list[FrameRecord], minimum: int, result: TrackResult | None = None) -> tuple[list[Candidate], int]:
        """Track the primary identity; raise AnalysisFailed with the training reason codes.
        ``result``: the tracker already ran over ``frames`` (DFD-FCG tracks during the pass)."""
        if not frames:
            raise AnalysisFailed("decode_failed")
        if not any(record.detected for record in frames):
            raise AnalysisFailed("no_face")
        if not any(record.candidates for record in frames):
            raise AnalysisFailed("low_detection_confidence")
        if result is None:
            result = track_candidates([record.candidates for record in frames], self.policy)
        if not result.candidates:
            raise AnalysisFailed("identity_track_failed")
        if len(result.candidates) < minimum:
            ambiguous = result.ambiguous_frames
            raise AnalysisFailed(
                "multiple_face_ambiguity"
                if ambiguous and len(result.candidates) + ambiguous >= minimum
                else "insufficient_valid_frames"
            )
        return result.candidates, result.ambiguous_frames


def phase_clips(track: list[Candidate], phases: int, clip_length: int, stride: int, max_gap: float) -> tuple[list[list[Candidate]], int]:
    """Tile each interleaved phase of the dense track into ``clip_length``-frame clips.

    Phase p holds the tracked frames whose dense position is p mod ``phases``, so its
    frames sit ``phases`` frames apart (the trained spacing). Each phase is split where
    the face disappears for longer than ``max_gap`` seconds and tiled with ``stride``,
    plus clips flush with both ends, so every tracked frame lands in at least one clip.
    Each phase's tiling starts at a different offset, which spreads the clip centres
    evenly through the video for the timeline at almost no extra cost.
    """
    clips: list[list[Candidate]] = []
    segments = 0
    for p in range(phases):
        sequence = [c for c in track if c.record.dense_pos % phases == p]
        start = 0
        for i in range(1, len(sequence) + 1):
            if i == len(sequence) or sequence[i].record.timestamp - sequence[i - 1].record.timestamp > max_gap:
                part = sequence[start:i]
                start = i
                if len(part) < clip_length:
                    continue
                segments += 1
                offset = round(p * stride / phases) if phases > 1 else 0
                last = len(part) - clip_length
                starts = sorted({0, last, *range(offset, last + 1, stride)})
                clips.extend(part[s : s + clip_length] for s in starts)
    return clips, segments


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class DetectorService:
    def __init__(self, registry_path: Path, weights_dir: Path, device: str = "auto", only: list[str] | None = None) -> None:
        cfg = json.loads(registry_path.read_text(encoding="utf-8"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # Same determinism settings as the training face cache.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.policy = cfg["policy"]
        self.whole = cfg["whole_video"]
        self.heatmap_cfg = cfg["heatmap"]
        self.inference_cfg = cfg["inference"]
        self.clip_length = int(self.policy["evidence"]["clip_length"])
        self.faces = FacePipeline(self.policy, self.device, {**self.inference_cfg, **self.whole,
                                                              "preview_width": self.heatmap_cfg["frame_width"]})
        self.fcg_cfg = cfg.get("dfd_fcg", {})
        self.models: dict[str, ModelEntry] = {}
        for m in cfg["models"]:
            if only and m["id"] not in only:
                continue
            arch = m.get("arch", "video_swin_b")
            if arch == "dfd_fcg":
                net, ckpt = dfd_fcg.load_dfd_fcg(weights_dir / m["checkpoint"], self.device)
            else:
                net, ckpt = load_model(weights_dir / m["checkpoint"], self.device)
            self.models[m["id"]] = ModelEntry(
                id=m["id"],
                name=m["name"],
                dataset=m["dataset"],
                temperature=float(m["temperature"]),
                threshold=float(m["threshold"]),
                frame_spacing_s=float(m["frame_spacing_s"]),
                net=net,
                epoch=ckpt.get("epoch"),
                metrics=ckpt.get("metrics"),
                arch=arch,
                clip_length=dfd_fcg.NUM_FRAMES if arch == "dfd_fcg" else self.clip_length,
            )
        if any(entry.arch == "dfd_fcg" for entry in self.models.values()):
            self.faces.landmarker = dfd_fcg.Landmarker(
                dfd_fcg.landmark_weights(weights_dir, self.fcg_cfg["landmarks"]["file"]), self.device)
        if self.device.type == "cuda":
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 2**30
            self.batch_size = int(self.inference_cfg["batch_size_cuda" if memory_gb >= 20 else "batch_size_cuda_small"])
            self.fcg_batch = int(self.fcg_cfg.get("batch_clips_cuda" if memory_gb >= 20 else "batch_clips_cuda_small", 4))
        else:
            self.amp_dtype = None
            self.batch_size = int(self.inference_cfg["batch_size_cpu"])
            self.fcg_batch = int(self.fcg_cfg.get("batch_clips_cpu", 1))
        lut = torch.from_numpy(HEAT_LUT.astype(np.float32)).to(self.device)
        self._lut_rgb, self._lut_alpha = lut[:, :3], lut[:, 3] / 255.0
        feather = _edge_feather(self.faces.size)
        self._feather = torch.from_numpy(feather.astype(np.float32)).to(self.device)
        # Several scans share the GPU (see ScanGate). The model pass, their largest memory step, runs
        # on ``heavy_slots`` shared streams (one by default), one scan at a time per stream.
        self.gate = ScanGate(scan_slots(self.device), self.device)
        heavy = int(os.environ.get("ORIGINAI_HEAVY_SLOTS", "0") or 0) or 1
        self.heavy_slots = min(heavy, self.gate.slots)
        self._heavy = threading.BoundedSemaphore(self.heavy_slots)
        self._heavy_streams: queue.Queue = queue.Queue()
        if self.device.type == "cuda":
            for _ in range(self.heavy_slots):
                self._heavy_streams.put(torch.cuda.Stream(self.device, priority=0))
        if self.device.type == "cuda":
            self._warm_up()

    # -- scoring -----------------------------------------------------------
    @contextlib.contextmanager
    def _model_pass(self):
        """Run the block on a shared model-pass stream (at most ``heavy_slots`` blocks at once).

        PyTorch caches freed GPU memory per stream, so a pass on every scan's own stream would keep
        its few GB of temporaries reserved once per scan. The pass stream waits for the caller's stream before
        starting, and the caller's stream waits for the pass after; tensors the pass returns on the
        GPU are handed to the caller's stream with ``_handover``."""
        with self._heavy:
            if self.device.type != "cuda":
                yield
                return
            stream = self._heavy_streams.get()
            caller = torch.cuda.current_stream(self.device)
            try:
                stream.wait_stream(caller)
                with torch.cuda.stream(stream):
                    yield
                caller.wait_stream(stream)
            finally:
                self._heavy_streams.put(stream)

    def _handover(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """A GPU tensor made in ``_model_pass``, now used on the caller's stream: tell the allocator
        so its memory is not reused by the pass stream while the caller still reads it."""
        if tensor is not None and tensor.is_cuda:
            tensor.record_stream(torch.cuda.current_stream(self.device))
        return tensor

    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    @torch.inference_mode()
    def _score(self, net: VideoSwinDetector, crops: torch.Tensor, clip_index: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        """Clip logits and their final-stage token contributions (K x 8 x 7 x 7, on device)."""
        last = len(net.backbone.features) - 1
        logits, maps = [], []
        with self._model_pass():
            for i in range(0, len(clip_index), self.batch_size):
                x = crops[clip_index[i : i + self.batch_size]].permute(0, 1, 4, 2, 3)  # B,T,C,H,W uint8
                with self._autocast():
                    activation = net.stage_activation(x, last)
                clip_logits, contributions = net.readout(activation)
                logits.append(clip_logits)
                maps.append(contributions)
            logits, maps = torch.cat(logits).double().cpu().numpy(), torch.cat(maps)
        return logits, self._handover(maps)

    @torch.inference_mode()
    def _score_fcg(self, net: dfd_fcg.DfdFcg, crops: torch.Tensor, clip_index: torch.Tensor,
                   evidence: bool = True) -> tuple[np.ndarray, np.ndarray, torch.Tensor | None]:
        """DFD-FCG clip scores: fake probabilities, their log-odds and, if asked for, each
        clip's evidence (K x 10 x 16 x 16, on device). ``clip_index`` (K, 10) holds rows of
        ``crops``, the 150 x 150 face crops. The image encoder runs in half precision on the
        GPU, as in the authors' evaluation."""
        probs, logits, maps = [], [], []
        with self._model_pass():
            for i in range(0, len(clip_index), self.fcg_batch):
                rows = clip_index[i : i + self.fcg_batch]
                x = dfd_fcg.prepare_frames(crops[rows.flatten()]).unflatten(0, tuple(rows.shape))
                with torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else contextlib.nullcontext():
                    out = net(x, evidence=evidence)
                probs.append(out["prob_fake"])
                logits.append(out["logit"])
                if evidence:
                    maps.append(out["evidence"])
            probs, logits = torch.cat(probs).double().cpu().numpy(), torch.cat(logits).double().cpu().numpy()
            maps = torch.cat(maps) if evidence else None
        return probs, logits, self._handover(maps)

    def _warm_up(self) -> None:
        """Compile/initialise every CUDA kernel once so the first request is not slow."""
        frames = torch.randint(0, 255, (2, 3, 360, 640), dtype=torch.uint8, device=self.device)
        detect_faces(frames, self.faces.min_face, self.faces.pnet, self.faces.rnet, self.faces.onet, self.faces.thresholds, self.faces.factor)
        crops = torch.randint(0, 255, (self.clip_length, self.faces.size, self.faces.size, 3), dtype=torch.uint8, device=self.device)
        self.faces._embed(crops[:2])
        index = torch.arange(self.clip_length, device=self.device)[None]
        for entry in [e for e in self.models.values() if e.arch == "video_swin_b"][:1]:
            self._score(entry.net, crops, index)
        for entry in [e for e in self.models.values() if e.arch == "dfd_fcg"][:1]:
            for _ in range(3):  # TorchScript profiles a shape on its first runs and optimises after
                self.faces.landmarker(frames, [0, 1], np.array([[200.0, 80.0, 440.0, 320.0]] * 2))
            faces = torch.randint(0, 255, (entry.clip_length, dfd_fcg.CROP_SIZE, dfd_fcg.CROP_SIZE, 3), dtype=torch.uint8, device=self.device)
            self._score_fcg(entry.net, faces, torch.arange(entry.clip_length, device=self.device)[None])
        _jpegs(crops[:1].permute(0, 3, 1, 2), 80)
        torch.cuda.synchronize()

    # -- evidence maps -------------------------------------------------------
    def _moment_frames(self, video: DenseVideo, entry: ModelEntry, clip: list[Candidate], slots: int,
                       face_crops: torch.Tensor | None, face_rows: dict[int, int] | None,
                       geometry: list[tuple[np.ndarray, int, int]] | None) -> tuple[list[Candidate], torch.Tensor, np.ndarray]:
        """The frames an evidence map is drawn on: each one's candidate, the face crop the
        model saw (S x size x size x 3 uint8) and its frame -> crop transform.

        Video Swin: one map per 2-frame tubelet, shown on the tubelet's first frame.
        DFD-FCG: one map per frame, on its 150 x 150 crop resized to the model's 224 x 224 input."""
        if entry.arch == "dfd_fcg":
            rows = [face_rows[id(c)] for c in clip]
            shown = face_crops[torch.tensor(rows, device=self.device)].permute(0, 3, 1, 2).float()
            shown = F.interpolate(shown, size=(self.faces.size, self.faces.size), mode="bicubic", align_corners=False, antialias=True)
            crops = shown.clamp_(0, 255).round_().to(torch.uint8).permute(0, 2, 3, 1)
            return clip, crops, np.stack([dfd_fcg.input_matrix(*geometry[row]) for row in rows])
        step = entry.clip_length // slots
        chosen = [clip[s * step] for s in range(slots)]
        crops = video.crops[torch.tensor([c.crop_index for c in chosen], device=self.device)]
        return chosen, crops, np.stack([c.matrix for c in chosen])

    @torch.inference_mode()
    def _render(self, video: DenseVideo, chosen: list[Candidate], relative: torch.Tensor, crops: torch.Tensor,
                mats: np.ndarray) -> list[dict]:
        """One picture set per evidence map, rendered on the GPU (Video Swin: eight per clip, the
        first frame of each 2-frame tubelet). ``crops`` and ``mats`` come from ``_moment_frames``.

        Each heat image is the frame with the evidence blended in; the page stacks it over the
        plain frame and its opacity slider mixes the two, which equals scaling the overlay alpha.
        """
        size = self.faces.size
        quality = int(self.heatmap_cfg["jpeg_quality"])
        heat = F.interpolate(relative[:, None].float(), size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)[:, 0]
        q = (heat * 255).round().long()
        rgb = self._lut_rgb[q].permute(0, 3, 1, 2)  # (S,3,size,size)
        alpha = (self._lut_alpha[q] * self._feather)[:, None]  # (S,1,size,size)

        crops = crops.permute(0, 3, 1, 2).float()
        crop_heat = (crops * (1 - alpha) + rgb * alpha).round().clamp(0, 255).to(torch.uint8)

        previews = video.previews[[c.record.preview_index for c in chosen]].to(self.device).float()
        ph, pw = previews.shape[2:]
        mats = torch.from_numpy(np.asarray(mats).astype(np.float64)).to(self.device)
        mats[:, :, :2] /= video.preview_scale  # preview pixels -> crop pixels
        ys, xs = torch.meshgrid(torch.arange(ph, device=self.device, dtype=torch.float64),
                                torch.arange(pw, device=self.device, dtype=torch.float64), indexing="ij")
        u = mats[:, 0, 0, None, None] * xs + mats[:, 0, 1, None, None] * ys + mats[:, 0, 2, None, None]
        v = mats[:, 1, 0, None, None] * xs + mats[:, 1, 1, None, None] * ys + mats[:, 1, 2, None, None]
        grid = torch.stack([u / (size - 1) * 2 - 1, v / (size - 1) * 2 - 1], dim=-1).float()
        rgba = torch.cat([rgb, alpha], dim=1)
        warped = F.grid_sample(rgba, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        a = warped[:, 3:4]
        frame_heat = (previews * (1 - a) + warped[:, :3] * a).round().clamp(0, 255).to(torch.uint8)

        frame_jpg = _jpegs(previews.round().to(torch.uint8), quality)
        frame_heat_jpg = _jpegs(frame_heat, quality)
        crop_jpg = _jpegs(crops.round().to(torch.uint8), quality)
        crop_heat_jpg = _jpegs(crop_heat, quality)
        peaks = relative.amax(dim=(1, 2)).float().cpu().numpy()
        return [
            {
                "t": round(c.record.timestamp, 3),
                "frame_jpg": _b64(frame_jpg[k]),
                "frame_heat_jpg": _b64(frame_heat_jpg[k]),
                "crop_jpg": _b64(crop_jpg[k]),
                "crop_heat_jpg": _b64(crop_heat_jpg[k]),
                "peak": round(float(peaks[k]), 3),
            }
            for k, c in enumerate(chosen)
        ]

    def _readout(self, entry: ModelEntry, logits: np.ndarray) -> dict:
        video_logit = float(np.mean(logits))
        calibrated = _sigmoid(video_logit / entry.temperature)
        return {
            "video_logit": video_logit,
            "prob_fake": calibrated,
            "raw_prob_fake": _sigmoid(video_logit),
            "pred": "DEEPFAKE" if calibrated >= entry.threshold else "REAL",
        }

    # -- public API ----------------------------------------------------------
    def analyze(self, video_path: Path, model_id: str) -> dict:
        entry = self.models.get(model_id)
        if entry is None:
            raise KeyError(model_id)
        try:
            with self.gate.slot():
                return self._analyze(video_path, entry)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            log.warning("Out of GPU memory next to other scans; re-running this scan on its own")
        gc.collect()  # the failed attempt's tensors
        with self.gate.slot(exclusive=True):
            torch.cuda.empty_cache()
            return self._analyze(video_path, entry)

    def status(self) -> dict:
        return {**self.gate.status(), "heavy_slots": self.heavy_slots}

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _readout_fcg(self, entry: ModelEntry, probs: np.ndarray) -> dict:
        """DFD-FCG's video score is the mean of its clips' fake probabilities (the authors' rule)."""
        mean = float(np.mean(probs))
        calibrated = _sigmoid(_logit(mean) / entry.temperature)
        return {
            "video_logit": _logit(mean),
            "prob_fake": calibrated,
            "raw_prob_fake": mean,
            "pred": "DEEPFAKE" if calibrated >= entry.threshold else "REAL",
        }

    def _score_video_fcg(self, video_path: Path, video: DenseVideo, entry: ModelEntry, dense: list[FrameRecord],
                         phases: int, spacing: float, timings: dict[str, float]) -> dict:
        """DFD-FCG on every frame: follow the face, cut its 150 x 150 crop from each frame, split
        the frames into interleaved phases at the trained spacing and tile each phase into
        10-frame clips. The authors' own evaluation (one clip per whole 3 seconds) is reported
        alongside as ``protocol``."""
        t = time.perf_counter()
        canvas = video.canvas
        # The tracker ran during the pass (same tracker, same checks); the crops were cut there too,
        # for the tracked faces that have landmarks and a crop transform (dfd_fcg.crop_geometry).
        _, ambiguous = self.faces.track(dense, entry.clip_length, result=canvas.tracker.result())
        track, geometry, crops = [c for c, _ in canvas.kept], [g for _, g in canvas.kept], canvas.crops
        if len(track) < entry.clip_length:
            raise AnalysisFailed("insufficient_valid_frames")
        timings["tracking"] = time.perf_counter() - t

        t = time.perf_counter()
        clips, segments = phase_clips(
            track, phases, entry.clip_length, int(self.fcg_cfg.get("clip_stride_in_phase", entry.clip_length)),
            float(self.whole["max_gap_factor"]) * max(spacing, entry.frame_spacing_s),
        )
        if not clips:
            raise AnalysisFailed("insufficient_valid_frames")
        self._sync()
        timings["face_crops"] = time.perf_counter() - t

        t = time.perf_counter()
        rows = {id(c): row for row, c in enumerate(track)}
        index = torch.tensor([[rows[id(c)] for c in clip] for clip in clips], device=self.device)
        probs, logits, maps = self._score_fcg(entry.net, crops, index)
        self._sync()
        timings["inference"] = time.perf_counter() - t

        # The authors' evaluation: one clip per whole 3 seconds, 10 frames spread evenly over it.
        t = time.perf_counter()
        by_frame = {c.record.frame_index: row for row, c in enumerate(track)}
        windows = dfd_fcg.protocol_frames(video.frames_decoded, video.fps or len(dense) / max(video.duration, 1e-6))
        usable = [w for w in windows if all(frame in by_frame for frame in w)]
        protocol: dict[str, Any] = {"available": False, "clip_count": len(usable), "windows": len(windows)}
        if usable:
            p_index = torch.tensor([[by_frame[frame] for frame in w] for w in usable], device=self.device)
            p_probs, _, _ = self._score_fcg(entry.net, crops, p_index, evidence=False)
            protocol.update(self._readout_fcg(entry, p_probs))
            protocol.update(available=True, clip_probs=[float(v) for v in p_probs], track_frames=len(track))
        else:
            protocol.update(reason="insufficient_valid_clips", message=FAILURE_MESSAGES["insufficient_valid_clips"])
        timings["protocol"] = time.perf_counter() - t
        return {
            "protocol": protocol, "clips": clips, "track": track, "ambiguous": ambiguous, "segments": segments,
            "fallback": None, "logits": logits, "probs": probs, "maps": maps, "readout": self._readout_fcg(entry, probs),
            "face_crops": crops, "face_rows": rows, "geometry": geometry,
        }

    def _analyze(self, video_path: Path, entry: ModelEntry) -> dict:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        hc = self.heatmap_cfg
        fcg = entry.arch == "dfd_fcg"
        video = self.faces.read(video_path, entry.frame_spacing_s, landmarks=fcg)
        self._sync()
        timings["decode_detect_align"] = time.perf_counter() - t0
        timings.update({f"stage_{k}": v for k, v in video.timings.items()})

        dense = [r for r in video.records if r.dense_pos >= 0]
        fps_eff = (video.fps or (len(dense) / max(video.duration, 1e-6))) / video.subsample
        phases = max(1, int(round(entry.frame_spacing_s * fps_eff)))
        spacing = phases / fps_eff
        if fcg:
            scored = self._score_video_fcg(video_path, video, entry, dense, phases, spacing, timings)
        else:
            scored = self._score_video_swin(video, entry, dense, phases, spacing, timings)
        protocol, clips, track, ambiguous = scored["protocol"], scored["clips"], scored["track"], scored["ambiguous"]
        segments, whole_failure, logits, probs = scored["segments"], scored["fallback"], scored["logits"], scored["probs"]
        contributions, readout = scored["maps"], scored["readout"]

        def summary(members: np.ndarray) -> tuple[float, float]:
            """Log-odds and calibrated probability of a group of clips, read out as the video is."""
            if probs is None:
                mean_logit = float(np.mean(logits[members]))
                return mean_logit, _sigmoid(mean_logit / entry.temperature)
            mean_logit = _logit(float(np.mean(probs[members])))
            return mean_logit, _sigmoid(mean_logit / entry.temperature)

        # Timeline: average the clips whose centre falls in each window of half a clip.
        starts = np.array([clip[0].record.timestamp for clip in clips])
        ends = np.array([clip[-1].record.timestamp for clip in clips])
        centres = (starts + ends) / 2
        span = float(np.median(ends - starts)) if len(clips) else 0.0
        width = max(span / 2, 1e-3)
        origin = float(starts.min())
        bucket = np.floor((centres - origin) / width).astype(int)
        timeline, bucket_to_segment = [], {}
        for b in sorted(set(bucket.tolist())):
            members = np.where(bucket == b)[0]
            mean_logit, mean_prob = summary(members)
            bucket_to_segment[b] = len(timeline)
            timeline.append(
                {
                    "index": len(timeline),
                    "start_s": round(origin + b * width, 3),
                    "end_s": round(min(origin + (b + 1) * width, float(ends.max())), 3),
                    "logit": round(mean_logit, 4),
                    "prob": round(mean_prob, 4),
                    "clips": int(len(members)),
                    "heatmap": False,
                }
            )

        # Evidence maps for the most suspicious clips (spread over the video) and the least.
        t = time.perf_counter()
        heatmaps = None
        top_k = int(hc["top_k"])
        if top_k > 0 and clips:
            order = [int(i) for i in np.argsort(-logits, kind="stable")]
            chosen: list[tuple[int, str]] = []
            used_buckets: set[int] = set()
            for i in order:
                if len(chosen) == top_k:
                    break
                if bucket[i] in used_buckets or any(abs(centres[i] - centres[j]) < span / 2 for j, _ in chosen):
                    continue
                chosen.append((i, "top"))
                used_buckets.add(int(bucket[i]))
            lowest = order[-1]
            if hc.get("include_lowest") and len(order) > len(chosen) and int(bucket[lowest]) not in used_buckets:
                chosen.append((lowest, "lowest"))
            sign = 1.0 if readout["pred"] == "DEEPFAKE" else -1.0
            maps = sign * contributions[torch.tensor([i for i, _ in chosen], device=self.device)].float()
            flat = maps.flatten()
            low = torch.quantile(flat, float(hc["low_percentile"]) / 100.0)
            high = torch.quantile(flat, float(hc["high_percentile"]) / 100.0)
            relative = ((maps - low) / (high - low)).clamp(0, 1) if high > low else torch.zeros_like(maps)
            rendered = []
            for (clip_i, role), rel in zip(chosen, relative):
                segment = bucket_to_segment[int(bucket[clip_i])]
                timeline[segment]["heatmap"] = True
                shown, shown_crops, shown_mats = self._moment_frames(
                    video, entry, clips[clip_i], rel.shape[0], scored.get("face_crops"), scored.get("face_rows"),
                    scored.get("geometry"))
                rendered.append({
                    "clip_index": segment,
                    "role": role,
                    "clip_start_s": round(float(starts[clip_i]), 3),
                    "clip_end_s": round(float(ends[clip_i]), 3),
                    "clip_prob": round(_sigmoid(float(logits[clip_i]) / entry.temperature), 4),
                    "frames": self._render(video, shown, rel, shown_crops, shown_mats),
                })
            heatmaps = {
                "method": ("Patch contributions to the DFD-FCG heads" if fcg
                           else "Class activation map (final Video Swin stage)"),
                "target": readout["pred"],
                "scale": "relative within this video",
                "frame_size": list(video.preview_size),
                "crop_size": [self.faces.size, self.faces.size],
                "stops": [[v, list(c)] for v, c in HEAT_STOPS],
                "clips": rendered,
            }
        timings["heatmaps"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - t0

        dense_frames = len(dense)
        return {
            "model": {"id": entry.id, "name": entry.name, "dataset": entry.dataset},
            **readout,
            "threshold": entry.threshold,
            "temperature": entry.temperature,
            "threshold_space": "calibrated",
            "aggregation": "mean_clip_probability" if fcg else "mean_clip_logit",
            "clips": timeline,
            "protocol": protocol,
            "coverage": {
                "mode": "every_frame_phased" if whole_failure is None else "protocol_fallback",
                "duration": round(video.duration, 3),
                "fps": video.fps,
                "width": video.width,
                "height": video.height,
                "rotation": video.rotation,
                "decoder": video.decoder,
                "device": str(self.device),
                "nvdec": nvdec_diagnostics() if self.device.type == "cuda" else None,
                "frames_decoded": video.frames_decoded,
                "frames_analyzed": dense_frames,
                "subsample": video.subsample,
                "face_frames": len(track),
                "ambiguous_frames": ambiguous,
                "phases": phases,
                "frame_spacing_s": round(spacing, 4),
                "trained_spacing_s": entry.frame_spacing_s,
                "clips": len(clips),
                "clip_length": entry.clip_length,
                "segments": segments,
                "timeline_segments": len(timeline),
                "analyzed_start_s": round(float(starts.min()), 3) if len(clips) else None,
                "analyzed_end_s": round(float(ends.max()), 3) if len(clips) else None,
                "fallback": whole_failure,
            },
            "meta": {"width": video.width, "height": video.height, "fps": video.fps, "duration": round(video.duration, 3)},
            "heatmaps": heatmaps,
            "timings": {k: round(v, 2) for k, v in timings.items()},
        }

    def _score_video_swin(self, video: DenseVideo, entry: ModelEntry, dense: list[FrameRecord], phases: int,
                          spacing: float, timings: dict[str, float]) -> dict:
        """Video Swin: the thesis protocol, then every frame in interleaved phases."""
        # Thesis protocol: 64 -> track -> 48 -> three anchors.
        t = time.perf_counter()
        evidence = self.policy["evidence"]
        protocol: dict[str, Any] = {"available": False, "clip_count": int(evidence["evaluation_clip_count"])}
        selected: list[Candidate] = []
        anchors: list[list[int]] = []
        try:
            p_track, _ = self.faces.track(video.protocol_frames, int(evidence["minimum_valid_frames"]))
            selected = [p_track[i] for i in deterministic_selection_indices(len(p_track), int(evidence["maximum_cached_frames"]))]
            anchors = evaluation_anchors(len(selected), self.clip_length, int(evidence["evaluation_clip_count"]))
            if len(anchors) != int(evidence["evaluation_clip_count"]):
                raise AnalysisFailed("insufficient_valid_clips")
            index = torch.tensor([[selected[i].crop_index for i in a] for a in anchors], device=self.device)
            logits, _ = self._score(entry.net, video.crops, index)
            protocol.update(self._readout(entry, logits))
            protocol.update(available=True, clip_logits=[float(v) for v in logits], track_frames=len(p_track))
        except AnalysisFailed as exc:
            protocol.update(reason=exc.reason, message=str(exc))

        timings["protocol"] = time.perf_counter() - t

        # Every frame: track the face through all analysed frames, split into phases at the
        # trained spacing, tile each phase into 16-frame clips.
        t = time.perf_counter()
        whole_failure = None
        try:
            track, ambiguous = self.faces.track(dense, self.clip_length)
            clips, segments = phase_clips(
                track, phases, self.clip_length, int(self.whole["clip_stride_in_phase"]),
                float(self.whole["max_gap_factor"]) * max(spacing, entry.frame_spacing_s),
            )
            if not clips:
                raise AnalysisFailed("insufficient_valid_frames")
        except AnalysisFailed as exc:
            if not protocol["available"]:
                raise
            whole_failure = exc.reason  # fall back to the protocol anchors
            track, ambiguous, segments = selected, 0, 1
            clips = [[selected[i] for i in a] for a in anchors]
        timings["tracking"] = time.perf_counter() - t

        t = time.perf_counter()
        index = torch.tensor([[c.crop_index for c in clip] for clip in clips], device=self.device)
        logits, contributions = self._score(entry.net, video.crops, index)
        self._sync()
        timings["inference"] = time.perf_counter() - t
        return {
            "protocol": protocol, "clips": clips, "track": track, "ambiguous": ambiguous, "segments": segments,
            "fallback": whole_failure, "logits": logits, "probs": None, "maps": contributions,
            "readout": self._readout(entry, logits),
        }

    def benchmark(self, model_id: str) -> dict:
        """Synthetic scoring pass used by the Vast PyWorker benchmark."""
        entry = self.models[model_id]
        with self.gate.slot():
            if entry.arch == "dfd_fcg":
                size, length = dfd_fcg.CROP_SIZE, entry.clip_length
                crops = torch.randint(0, 255, (length * 2, size, size, 3), dtype=torch.uint8, device=self.device)
                t = time.perf_counter()
                _, logits, _ = self._score_fcg(entry.net, crops, torch.arange(length * 2, device=self.device).view(2, length))
                self._sync()
                return {"seconds": time.perf_counter() - t, "logits": [float(v) for v in logits]}
            size = self.faces.size
            crops = torch.randint(0, 255, (self.clip_length * 2, size, size, 3), dtype=torch.uint8, device=self.device)
            index = torch.arange(self.clip_length * 2, device=self.device).view(2, self.clip_length)
            t = time.perf_counter()
            logits, _ = self._score(entry.net, crops, index)
            self._sync()
            return {"seconds": time.perf_counter() - t, "logits": [float(v) for v in logits]}
