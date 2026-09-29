"""Video Swin-B deepfake detector: thesis preprocessing, whole-video scoring, evidence maps.

Preprocessing is a port of the training repository
(Thesis_Code/src/swinfusionpp/preprocessing.py, policy ``preprocessing_v001``):

    PyAV decode at requested timestamps (first frame at or after each target)
    -> MTCNN, keep detections with probability >= 0.90
    -> five-landmark similarity alignment to the ArcFace template, 224x224 RGB uint8
    -> FaceNet (vggface2) identity tracking of the primary face.

Each video is scored twice with the selected checkpoint:

* ``protocol`` - the exact thesis evaluation: 64 uniform timestamps, 48 tracked
  frames, three 16-frame anchors ([0:16], [16:32], [32:48]).
* ``whole`` - timestamps every ``frame_spacing_s`` seconds (the median frame spacing
  the model saw in training) across the whole video, tiled into overlapping
  16-frame clips. Its mean clip logit drives the verdict.

Both follow the thesis read-out: mean clip logit -> sigmoid(logit / T) -> threshold.

Evidence maps: the Video Swin-B head is LayerNorm -> mean pool -> linear, so every
final-stage token (8 x 7 x 7 per clip) contributes exactly ``w . LayerNorm(token)`` to
the clip logit, and the mean of those contributions plus the bias *is* the logit.
This class activation map (equivalent to Grad-CAM taken at the final norm layer) comes
free with the scoring pass. It is shown relative to the video: the strongest evidence
for the verdict is brightest.
"""

from __future__ import annotations

import base64
import contextlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models.video import swin3d_b

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
    net: VideoSwinDetector
    epoch: int | None = None
    metrics: dict | None = None

    def public_info(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
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
    """One decoded frame. Several requested timestamps may share a record."""

    timestamp: float
    frame_index: int
    detected: bool = False
    candidates: list["Candidate"] = field(default_factory=list)
    preview_jpeg: bytes | None = None


@dataclass(frozen=True)
class Candidate:
    frame_position: int
    box: tuple[float, float, float, float]
    probability: float
    landmarks: tuple[tuple[float, float], ...]
    crop: np.ndarray
    embedding: np.ndarray
    # Additions for the web service (unused by tracking):
    matrix: np.ndarray  # full-frame -> crop similarity transform
    record: FrameRecord


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


def align_face_with_matrix(rgb: np.ndarray, landmarks: np.ndarray, size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    matrix = similarity_transform(np.asarray(landmarks), canonical_landmarks(size))
    crop = cv2.warpAffine(
        rgb,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return crop, matrix


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


def track_candidates(candidates_by_frame: list[list[Candidate]], policy: dict[str, Any]) -> TrackResult:
    settings = policy["tracking"]
    maximum_distance = float(settings["maximum_cosine_distance"])
    minimum_iou = float(settings["minimum_iou_for_spatial_fallback"])
    ambiguity_margin = float(settings["ambiguity_margin"])
    ema_weight = float(settings["embedding_ema"])
    track: list[Candidate] = []
    association_costs: list[float] = []
    reference_embedding: np.ndarray | None = None
    previous_box: tuple[float, float, float, float] | None = None
    ambiguous = 0
    for candidates in candidates_by_frame:
        if not candidates:
            continue
        if reference_embedding is None:
            selected = max(candidates, key=lambda item: item.probability)
            selected_cost = 0.0
        else:
            scored = []
            for candidate in candidates:
                distance = cosine_distance(reference_embedding, candidate.embedding)
                iou = box_iou(previous_box, candidate.box) if previous_box else 0.0
                if distance <= maximum_distance or iou >= minimum_iou:
                    scored.append((distance + 0.15 * (1.0 - iou), distance, candidate))
            scored.sort(key=lambda item: (item[0], -item[2].probability))
            if not scored:
                continue
            if len(scored) > 1 and scored[1][0] - scored[0][0] < ambiguity_margin:
                ambiguous += 1
                continue
            if scored[0][1] > maximum_distance and box_iou(previous_box, scored[0][2].box) < minimum_iou:
                continue
            selected = scored[0][2]
            selected_cost = float(scored[0][0])
        track.append(selected)
        association_costs.append(selected_cost)
        if reference_embedding is None:
            reference_embedding = selected.embedding.astype(np.float32, copy=True)
        else:
            reference_embedding = ema_weight * reference_embedding + (1.0 - ema_weight) * selected.embedding
            norm = np.linalg.norm(reference_embedding)
            if norm:
                reference_embedding /= norm
        previous_box = selected.box
    gaps = len(candidates_by_frame) - len(track)
    return TrackResult(
        candidates=track,
        association_costs=association_costs,
        ambiguous_frames=ambiguous,
        gap_frames=gaps,
        termination_reason=("completed_candidate_sequence" if track else "no_primary_identity_initialized"),
    )


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


def detection_batch_size(width: int, height: int, policy: int | dict[str, int]) -> int:
    if isinstance(policy, dict):
        return int(
            policy["high_resolution"]
            if width * height >= int(policy["high_resolution_min_pixels"])
            else policy["default"]
        )
    return int(policy)


def _stream_duration(container, stream) -> float:
    """Same order as the training inventory/decoder."""
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    if stream.frames and stream.average_rate:
        return float(stream.frames / stream.average_rate)
    raise AnalysisFailed("metadata_error")


# ---------------------------------------------------------------------------
# Whole-video tiling
# ---------------------------------------------------------------------------
def tile_windows(times: list[float], clip_length: int, stride: int, max_gap: float) -> tuple[list[int], int]:
    """Start indices of overlapping ``clip_length`` windows over the ordered track.

    The track is split where consecutive frames are more than ``max_gap`` seconds
    apart (the face left the shot), so a clip never bridges a long gap. Each segment
    is covered end to end: stride ``stride`` plus a final window flush with its end.
    """
    segments, begin = [], 0
    for i in range(1, len(times) + 1):
        if i == len(times) or times[i] - times[i - 1] > max_gap:
            segments.append((begin, i))
            begin = i
    if not any(end - start >= clip_length for start, end in segments) and len(times) >= clip_length:
        segments = [(0, len(times))]  # sparse faces: tile across gaps, as training did
    starts: list[int] = []
    used = 0
    for start, end in segments:
        if end - start < clip_length:
            continue
        used += 1
        seg = list(range(start, end - clip_length + 1, stride))
        if seg[-1] != end - clip_length:
            seg.append(end - clip_length)
        starts.extend(seg)
    return starts, used


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


def _jpeg(rgb: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _png_rgba(rgba: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Face pipeline
# ---------------------------------------------------------------------------
@dataclass
class DecodedVideo:
    duration: float
    fps: float | None
    width: int
    height: int
    rotation: int
    preview_size: tuple[int, int]
    preview_scale: float
    protocol_frames: list[FrameRecord]
    whole_frames: list[FrameRecord]
    unique_frames: int


class FacePipeline:
    def __init__(self, policy: dict[str, Any], device: torch.device) -> None:
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.policy = policy
        self.device = device
        detection = policy["face_detection"]
        self.detector = MTCNN(
            keep_all=True,
            select_largest=False,
            min_face_size=int(detection["min_face_size"]),
            thresholds=list(map(float, detection["thresholds"])),
            factor=float(detection["factor"]),
            device=device,
            post_process=False,
        )
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self.embedder.requires_grad_(False)
        self.size = int(policy["alignment"]["output_height"])

    def _embed(self, crops: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        from facenet_pytorch import fixed_image_standardization

        if not crops:
            return np.empty((0, 512), dtype=np.float32)
        tensors = []
        for crop in crops:
            resized = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float()
            tensors.append(fixed_image_standardization(tensor))
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[start : start + batch_size]).to(self.device)
                outputs.append(self.embedder(batch).cpu().numpy())
        return np.concatenate(outputs).astype(np.float32)

    def _detect(self, batch: list[tuple[FrameRecord, np.ndarray]], pending: list) -> None:
        minimum_confidence = float(self.policy["face_detection"]["minimum_confidence"])
        images = [Image.fromarray(rgb) for _, rgb in batch]
        if len({image.size for image in images}) == 1:
            boxes, probabilities, landmarks = self.detector.detect(images, landmarks=True)
        else:  # resolution changed mid-stream; MTCNN batches need equal sizes
            results = [self.detector.detect([image], landmarks=True) for image in images]
            boxes, probabilities, landmarks = ([r[k][0] for r in results] for k in range(3))
        for (record, rgb), frame_boxes, frame_probabilities, frame_landmarks in zip(
            batch, boxes, probabilities, landmarks, strict=True
        ):
            if frame_boxes is None:
                continue
            record.detected = True
            for box, probability, points in zip(frame_boxes, frame_probabilities, frame_landmarks, strict=True):
                if not detection_is_accepted(probability, minimum_confidence):
                    continue
                crop, matrix = align_face_with_matrix(rgb, np.asarray(points), self.size)
                pending.append((record, tuple(map(float, box)), float(probability), np.asarray(points), crop, matrix))

    def read(self, path: Path, spacing: float, preview_width: int, preview_quality: int) -> DecodedVideo:
        """Decode the protocol and whole-video timestamps in one pass, detect and align faces.

        Full-resolution frames are released after detection; only aligned crops and
        small preview JPEGs are kept.
        """
        decoder = self.policy["decoder"]
        edge = float(decoder["edge_fraction"])
        try:
            container = av.open(str(path), mode="r", metadata_errors="ignore")
        except Exception:
            raise AnalysisFailed("decode_failed")
        pending: list = []
        with container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise AnalysisFailed("decode_failed")
            stream.thread_type = "AUTO"
            duration = _stream_duration(container, stream)
            if duration <= 0:
                raise AnalysisFailed("metadata_error")
            protocol_requests = uniform_timestamps(duration, int(decoder["candidate_frame_count"]), edge)
            start, stop = duration * edge, max(duration * edge, duration * (1.0 - edge))
            whole_requests = np.arange(start, stop + 1e-9, spacing, dtype=np.float64)
            requests = np.concatenate([protocol_requests, whole_requests])
            order = np.argsort(requests, kind="stable")
            n_protocol = len(protocol_requests)
            assigned: list[FrameRecord | None] = [None] * len(requests)

            fps = float(stream.average_rate) if stream.average_rate else None
            batch_policy = self.policy["face_detection"].get("inference_batch_size", 16)
            batch: list[tuple[FrameRecord, np.ndarray]] = []
            width = height = rotation = 0
            preview_size, preview_scale = (0, 0), 1.0
            detection_batch = 16
            unique = 0
            target = 0
            try:
                for decoded_index, frame in enumerate(container.decode(stream)):
                    if frame.pts is not None and frame.time_base is not None:
                        timestamp = float(frame.pts * frame.time_base)
                    elif fps:
                        timestamp = decoded_index / fps
                    else:
                        continue
                    if target >= len(order) or timestamp + 1e-9 < requests[order[target]]:
                        continue
                    record = FrameRecord(timestamp=timestamp, frame_index=decoded_index)
                    while target < len(order) and timestamp + 1e-9 >= requests[order[target]]:
                        assigned[order[target]] = record
                        target += 1
                    rgb = frame.to_ndarray(format="rgb24")
                    turn = int(round(getattr(frame, "rotation", 0) or 0) / 90) % 4
                    if turn:  # phone videos: apply the display rotation
                        rgb = np.ascontiguousarray(np.rot90(rgb, turn))
                    if not width:
                        height, width = rgb.shape[:2]
                        rotation = turn * 90
                        detection_batch = detection_batch_size(width, height, batch_policy)
                        pw = min(int(preview_width), width)
                        preview_scale = pw / width
                        preview_size = (pw, max(2, int(round(height * preview_scale))))
                    small = cv2.resize(rgb, preview_size, interpolation=cv2.INTER_AREA) if preview_scale < 1 else rgb
                    record.preview_jpeg = _jpeg(small, preview_quality)
                    batch.append((record, rgb))
                    unique += 1
                    if len(batch) >= detection_batch:
                        self._detect(batch, pending)
                        batch = []
                    if target == len(order):
                        break
            except AnalysisFailed:
                raise
            except Exception:
                if not unique:
                    raise AnalysisFailed("decode_failed")
            if batch:
                self._detect(batch, pending)
        if not unique:
            raise AnalysisFailed("decode_failed")

        embeddings = self._embed([item[4] for item in pending])
        for (record, box, probability, points, crop, matrix), embedding in zip(pending, embeddings, strict=True):
            record.candidates.append(
                Candidate(
                    frame_position=record.frame_index,
                    box=box,
                    probability=probability,
                    landmarks=tuple(tuple(map(float, point)) for point in points),
                    crop=crop,
                    embedding=embedding,
                    matrix=matrix,
                    record=record,
                )
            )
        return DecodedVideo(
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            rotation=rotation,
            preview_size=preview_size,
            preview_scale=preview_scale,
            protocol_frames=[r for r in assigned[:n_protocol] if r is not None],
            whole_frames=[r for r in assigned[n_protocol:] if r is not None],
            unique_frames=unique,
        )

    def track(self, frames: list[FrameRecord], minimum: int) -> tuple[list[Candidate], int]:
        """Track the primary identity; raise AnalysisFailed with the training reason codes."""
        if not frames:
            raise AnalysisFailed("decode_failed")
        if not any(record.detected for record in frames):
            raise AnalysisFailed("no_face")
        if not any(record.candidates for record in frames):
            raise AnalysisFailed("low_detection_confidence")
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


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


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
        self._feather = _edge_feather(int(self.policy["alignment"]["output_height"]))
        self.inference_cfg = cfg["inference"]
        evidence = self.policy["evidence"]
        self.clip_length = int(evidence["clip_length"])
        self.faces = FacePipeline(self.policy, self.device)
        self.models: dict[str, ModelEntry] = {}
        for m in cfg["models"]:
            if only and m["id"] not in only:
                continue
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
            )
        if self.device.type == "cuda":
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.batch_size = int(self.inference_cfg["batch_size_cuda"])
        else:
            self.amp_dtype = None
            self.batch_size = int(self.inference_cfg["batch_size_cpu"])
        # One GPU, one request at a time keeps memory predictable.
        self._lock = threading.Lock()

    # -- scoring -----------------------------------------------------------
    def _clip_tensor(self, track: list[Candidate], starts: list[int]) -> torch.Tensor:
        clips = np.stack([np.stack([track[s + j].crop for j in range(self.clip_length)]) for s in starts])
        return torch.from_numpy(clips).to(self.device).permute(0, 1, 4, 2, 3)  # B,T,C,H,W uint8

    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    def _score(self, net: VideoSwinDetector, track: list[Candidate], starts: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Clip logits and their final-stage token contributions (K x 8 x 7 x 7)."""
        last = len(net.backbone.features) - 1
        logits, maps = [], []
        for i in range(0, len(starts), self.batch_size):
            x = self._clip_tensor(track, starts[i : i + self.batch_size])
            with torch.inference_mode():
                with self._autocast():
                    activation = net.stage_activation(x, last)
                clip_logits, contributions = net.readout(activation)
            logits.append(clip_logits.cpu())
            maps.append(contributions.cpu())
        return torch.cat(logits).numpy().astype(np.float64), torch.cat(maps).numpy()

    def _render(self, video: DecodedVideo, track: list[Candidate], start: int, relative: np.ndarray) -> list[dict]:
        """Eight frames per clip: the first frame of each 2-frame tubelet (T' = 8)."""
        size = self.faces.size
        quality = int(self.heatmap_cfg["jpeg_quality"])
        pw, ph = video.preview_size
        slots = relative.shape[0]
        step = self.clip_length // slots
        frames = []
        for slot in range(slots):
            candidate = track[start + slot * step]
            value = relative[slot]
            heat = np.clip(cv2.resize(value.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
            rgba = HEAT_LUT[(heat * 255.0).round().astype(np.uint8)].copy()
            rgba[..., 3] = (rgba[..., 3] * self._feather).astype(np.uint8)
            matrix = candidate.matrix.astype(np.float64).copy()
            matrix[:, :2] /= video.preview_scale  # preview pixels -> crop pixels
            frame_heat = cv2.warpAffine(
                rgba,
                matrix,
                (pw, ph),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            frames.append(
                {
                    "t": round(candidate.record.timestamp, 3),
                    "frame_jpg": _b64(candidate.record.preview_jpeg or b""),
                    "frame_heat_png": _b64(_png_rgba(frame_heat)),
                    "crop_jpg": _b64(_jpeg(candidate.crop, quality)),
                    "crop_heat_png": _b64(_png_rgba(rgba)),
                    "peak": round(float(value.max()), 3),
                }
            )
        return frames

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
        with self._lock:
            return self._analyze(video_path, entry)

    def _analyze(self, video_path: Path, entry: ModelEntry) -> dict:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        hc = self.heatmap_cfg
        video = self.faces.read(video_path, entry.frame_spacing_s, int(hc["frame_width"]), int(hc["jpeg_quality"]))
        timings["decode_and_faces"] = time.perf_counter() - t0

        # Thesis protocol: 64 -> track -> 48 -> three anchors.
        t = time.perf_counter()
        evidence = self.policy["evidence"]
        protocol: dict[str, Any] = {"available": False, "clip_count": int(evidence["evaluation_clip_count"])}
        try:
            track, _ = self.faces.track(video.protocol_frames, int(evidence["minimum_valid_frames"]))
            selected = [track[i] for i in deterministic_selection_indices(len(track), int(evidence["maximum_cached_frames"]))]
            anchors = evaluation_anchors(len(selected), self.clip_length, int(evidence["evaluation_clip_count"]))
            if len(anchors) != int(evidence["evaluation_clip_count"]):
                raise AnalysisFailed("insufficient_valid_clips")
            logits, _ = self._score(entry.net, selected, [a[0] for a in anchors])
            protocol.update(self._readout(entry, logits))
            protocol.update(available=True, clip_logits=[float(v) for v in logits], track_frames=len(track))
        except AnalysisFailed as exc:
            protocol.update(reason=exc.reason, message=str(exc))
            selected, anchors = [], []

        # Whole video: tile the tracked face at the training frame spacing.
        whole_failure = None
        try:
            track, ambiguous = self.faces.track(video.whole_frames, self.clip_length)
            times = [c.record.timestamp for c in track]
            starts, segments = tile_windows(
                times,
                self.clip_length,
                int(self.whole["clip_stride"]),
                float(self.whole["max_gap_factor"]) * entry.frame_spacing_s,
            )
            if not starts:
                raise AnalysisFailed("insufficient_valid_frames")
        except AnalysisFailed as exc:
            if not protocol["available"]:
                raise
            whole_failure = exc.reason  # fall back to the protocol anchors
            track, ambiguous, segments = selected, 0, 1
            starts = [a[0] for a in anchors]
            times = [c.record.timestamp for c in track]
        logits, contributions = self._score(entry.net, track, starts)
        timings["inference"] = time.perf_counter() - t
        readout = self._readout(entry, logits)

        clips = []
        for index, (s, logit) in enumerate(zip(starts, logits)):
            clips.append(
                {
                    "index": index,
                    "start_s": round(times[s], 3),
                    "end_s": round(times[s + self.clip_length - 1], 3),
                    "logit": round(float(logit), 4),
                    "prob": round(_sigmoid(float(logit) / entry.temperature), 4),
                    "heatmap": False,
                }
            )

        # Evidence maps for the most suspicious clips (+ the least suspicious for contrast),
        # oriented toward the verdict and scaled together so clips stay comparable.
        t = time.perf_counter()
        heatmaps = None
        top_k = int(hc["top_k"])
        if top_k > 0 and clips:
            ranked = [int(i) for i in np.argsort(-logits, kind="stable")]
            chosen = [(i, "top") for i in ranked[:top_k]]
            if hc.get("include_lowest") and len(ranked) > top_k:
                chosen.append((ranked[-1], "lowest"))
            sign = 1.0 if readout["pred"] == "DEEPFAKE" else -1.0
            maps = sign * contributions[[i for i, _ in chosen]]
            low = float(np.percentile(maps, float(hc["low_percentile"])))
            high = float(np.percentile(maps, float(hc["high_percentile"])))
            relative = np.clip((maps - low) / (high - low), 0.0, 1.0) if high > low else np.zeros_like(maps)
            rendered = []
            for (clip_index, role), rel in zip(chosen, relative):
                clips[clip_index]["heatmap"] = True
                rendered.append(
                    {
                        "clip_index": clip_index,
                        "role": role,
                        "frames": self._render(video, track, starts[clip_index], rel),
                    }
                )
            heatmaps = {
                "method": "Class activation map (final Video Swin stage)",
                "target": readout["pred"],
                "scale": "relative within this video",
                "frame_size": list(video.preview_size),
                "crop_size": [self.faces.size, self.faces.size],
                "stops": [[v, list(c)] for v, c in HEAT_STOPS],
                "clips": rendered,
            }
        timings["heatmaps"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - t0

        return {
            "model": {"id": entry.id, "name": entry.name, "dataset": entry.dataset},
            **readout,
            "threshold": entry.threshold,
            "temperature": entry.temperature,
            "threshold_space": "calibrated",
            "aggregation": "mean_clip_logit",
            "clips": clips,
            "protocol": protocol,
            "coverage": {
                "duration": round(video.duration, 3),
                "fps": video.fps,
                "width": video.width,
                "height": video.height,
                "rotation": video.rotation,
                "frame_spacing_s": entry.frame_spacing_s,
                "frames_requested": len(video.whole_frames),
                "frames_decoded": video.unique_frames,
                "face_frames": len(track),
                "ambiguous_frames": ambiguous,
                "segments": segments,
                "clips": len(clips),
                "clip_length": self.clip_length,
                "clip_stride": int(self.whole["clip_stride"]),
                "analyzed_start_s": clips[0]["start_s"] if clips else None,
                "analyzed_end_s": max(c["end_s"] for c in clips) if clips else None,
                "fallback": whole_failure,
            },
            "meta": {"width": video.width, "height": video.height, "fps": video.fps, "duration": round(video.duration, 3)},
            "heatmaps": heatmaps,
            "timings": {k: round(v, 2) for k, v in timings.items()},
        }

    def benchmark(self, model_id: str) -> dict:
        """Synthetic scoring pass used by the Vast PyWorker benchmark."""
        entry = self.models[model_id]
        size = self.faces.size
        rng = np.random.default_rng(0)
        fake_track = [
            Candidate(0, (0, 0, 1, 1), 1.0, (), rng.integers(0, 255, (size, size, 3), dtype=np.uint8), np.zeros(512, np.float32), np.eye(2, 3, dtype=np.float32), FrameRecord(0.0, 0))
            for _ in range(self.clip_length * 2)
        ]
        t = time.perf_counter()
        with self._lock:
            logits, _ = self._score(entry.net, fake_track, [0, self.clip_length])
        return {"seconds": time.perf_counter() - t, "logits": [float(v) for v in logits]}
