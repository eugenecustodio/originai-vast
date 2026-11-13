"""SwinFusion++ for inference: Swin-B (spatial, per frame) + Video Swin-B (temporal, per clip), fused at all four
stages by TLBG-HCA (tubelet-aligned, local, bidirectional, gated hierarchical cross-attention).

The modules below are the thesis's own model code (Thesis_Code/src/swinfusionpp/models: backbones.py, fusion.py,
principal.py), kept layer for layer so the trained checkpoints load with ``strict=True``. Only what inference does
not need was left out: loading ImageNet/Kinetics backbone weights (the checkpoint holds every weight) and the
progressive-unfreezing helpers used in training.

Evidence maps. The fusion head pools every stage (mean and max) and ends in a two-layer classifier, so a clip's
logit does not split exactly into one value per token as it does for a single Video Swin-B. The maps are therefore
Grad-CAM (Selvaraju et al., 2017) on the stage-4 features of both encoders, the 7 x 7 grids the fusion's last stage
reads: each channel is weighted by the mean gradient of the clip logit with respect to it, and the weighted sum is
the map. The spatial encoder's 16 frames are averaged in pairs onto Video Swin's 8 tubelets, and the two maps are
added. Positive values pushed the clip toward Deepfake.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models import swin_b
from torchvision.models.video import swin3d_b

STAGE_CHANNELS = (128, 256, 512, 1024)
SPATIAL_STAGE_INDICES = (1, 3, 5, 7)
TEMPORAL_STAGE_INDICES = (0, 2, 4, 6)
CLIP_FRAMES = 16
SIZE = 224


# ---------------------------------------------------------------------------
# Backbones (thesis backbones.py)
# ---------------------------------------------------------------------------
class InputNormalizer(nn.Module):
    """RGB normalization for B,T,C,H,W clips; mean and std come from the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 1, 3, 1, 1), persistent=True)
        self.register_buffer("std", torch.ones(1, 1, 3, 1, 1), persistent=True)

    def forward(self, clip: Tensor) -> Tensor:
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"Expected B,T,3,H,W input, got {tuple(clip.shape)}")
        if not clip.is_floating_point():
            clip = clip.float().div_(255.0)
        return (clip - self.mean.to(clip)) / self.std.to(clip)


class SpatialSwinBackbone(nn.Module):
    """Torchvision Swin-B applied to every frame, with taps after the last block of each stage."""

    def __init__(self) -> None:
        super().__init__()
        model = swin_b(weights=None)
        self.features = model.features
        self.norm = model.norm
        self.avgpool = model.avgpool

    def forward(self, clip: Tensor) -> list[Tensor]:
        if clip.ndim != 5 or clip.shape[1:] != (CLIP_FRAMES, 3, SIZE, SIZE):
            raise ValueError(f"Expected B,16,3,224,224 input, got {tuple(clip.shape)}")
        batch, frames = clip.shape[:2]
        value = clip.reshape(batch * frames, 3, SIZE, SIZE)
        taps: list[Tensor] = []
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in SPATIAL_STAGE_INDICES:
                height, width, channels = value.shape[1:]
                taps.append(value.reshape(batch, frames, height, width, channels).permute(0, 1, 4, 2, 3).contiguous())
        return taps  # the pooled readout (norm, avgpool) is not used by SwinFusion++


class TemporalSwinBackbone(nn.Module):
    """Torchvision Video Swin-B with taps after the last block of each stage."""

    def __init__(self) -> None:
        super().__init__()
        model = swin3d_b(weights=None)
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.features = model.features
        self.norm = model.norm
        self.avgpool = model.avgpool

    def forward(self, clip: Tensor) -> list[Tensor]:
        if clip.ndim != 5 or clip.shape[1:] != (CLIP_FRAMES, 3, SIZE, SIZE):
            raise ValueError(f"Expected B,16,3,224,224 input, got {tuple(clip.shape)}")
        value = self.pos_drop(self.patch_embed(clip.permute(0, 2, 1, 3, 4)))
        taps: list[Tensor] = []
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in TEMPORAL_STAGE_INDICES:
                taps.append(value.permute(0, 1, 4, 2, 3).contiguous())
        return taps


# ---------------------------------------------------------------------------
# TLBG-HCA fusion (thesis fusion.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowShape:
    batch: int
    temporal: int
    height: int
    width: int
    channels: int
    window_size: int


def window_partition(value: Tensor, window_size: int = 7) -> tuple[Tensor, WindowShape]:
    batch, temporal, height, width, channels = value.shape
    windows = (
        value.reshape(batch, temporal, height // window_size, window_size, width // window_size, window_size, channels)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .reshape(-1, window_size * window_size, channels)
    )
    return windows, WindowShape(batch, temporal, height, width, channels, window_size)


def window_reverse(windows: Tensor, shape: WindowShape) -> Tensor:
    size = shape.window_size
    return (
        windows.reshape(shape.batch, shape.temporal, shape.height // size, shape.width // size, size, size, shape.channels)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .reshape(shape.batch, shape.temporal, shape.height, shape.width, shape.channels)
    )


class FeedForward(nn.Module):
    def __init__(self, dimensions: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimensions, 4 * dimensions), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dimensions, dimensions), nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class StageFusion(nn.Module):
    """One temporally aligned, bidirectional, gated cross-attention stage."""

    def __init__(self, input_channels: int, dimensions: int, heads: int, window_size: int = 7, dropout: float = 0.10) -> None:
        super().__init__()
        self.window_size = window_size
        self.temporal_adapter = nn.Conv3d(input_channels, input_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1),
                                          groups=input_channels)
        self.spatial_projection = nn.Linear(input_channels, dimensions)
        self.temporal_projection = nn.Linear(input_channels, dimensions)
        self.spatial_attention_norm = nn.LayerNorm(dimensions)
        self.temporal_attention_norm = nn.LayerNorm(dimensions)
        self.spatial_from_temporal = nn.MultiheadAttention(dimensions, heads, dropout=dropout, batch_first=True)
        self.temporal_from_spatial = nn.MultiheadAttention(dimensions, heads, dropout=dropout, batch_first=True)
        self.spatial_attention_gate = nn.Parameter(torch.zeros(()))
        self.temporal_attention_gate = nn.Parameter(torch.zeros(()))
        self.spatial_ffn_norm = nn.LayerNorm(dimensions)
        self.temporal_ffn_norm = nn.LayerNorm(dimensions)
        self.spatial_ffn = FeedForward(dimensions, dropout)
        self.temporal_ffn = FeedForward(dimensions, dropout)
        self.spatial_ffn_gate = nn.Parameter(torch.zeros(()))
        self.temporal_ffn_gate = nn.Parameter(torch.zeros(()))
        evidence_dimensions = 4 * dimensions
        self.evidence_norm = nn.LayerNorm(evidence_dimensions)
        self.readout = nn.Linear(2 * evidence_dimensions, 256)

    def forward(self, spatial: Tensor, temporal: Tensor) -> Tensor:
        spatial = self.temporal_adapter(spatial.permute(0, 2, 1, 3, 4)).permute(0, 2, 3, 4, 1)
        temporal = temporal.permute(0, 1, 3, 4, 2)
        spatial = self.spatial_projection(spatial)
        temporal = self.temporal_projection(temporal)
        spatial_windows, shape = window_partition(spatial, self.window_size)
        temporal_windows, _ = window_partition(temporal, self.window_size)
        spatial_norm = self.spatial_attention_norm(spatial_windows)
        temporal_norm = self.temporal_attention_norm(temporal_windows)
        spatial_update = self.spatial_from_temporal(spatial_norm, temporal_norm, temporal_norm, need_weights=False)[0]
        temporal_update = self.temporal_from_spatial(temporal_norm, spatial_norm, spatial_norm, need_weights=False)[0]
        spatial_enhanced = spatial_windows + self.spatial_attention_gate * spatial_update
        temporal_enhanced = temporal_windows + self.temporal_attention_gate * temporal_update
        spatial_enhanced = spatial_enhanced + self.spatial_ffn_gate * self.spatial_ffn(self.spatial_ffn_norm(spatial_enhanced))
        temporal_enhanced = temporal_enhanced + self.temporal_ffn_gate * self.temporal_ffn(self.temporal_ffn_norm(temporal_enhanced))
        spatial_grid = window_reverse(spatial_enhanced, shape)
        temporal_grid = window_reverse(temporal_enhanced, shape)
        evidence = torch.cat((spatial_grid, temporal_grid, (spatial_grid - temporal_grid).abs(), spatial_grid * temporal_grid), dim=-1)
        evidence = self.evidence_norm(evidence)
        return self.readout(torch.cat((evidence.mean(dim=(1, 2, 3)), evidence.amax(dim=(1, 2, 3))), dim=-1))


class TLBGHCA(nn.Module):
    def __init__(self, dropout: float = 0.10, classifier_dropout: float = 0.30) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            StageFusion(channel, dimension, head, dropout=dropout)
            for channel, dimension, head in zip(STAGE_CHANNELS, (64, 128, 256, 256), (2, 4, 8, 8), strict=True)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 512), nn.GELU(), nn.Dropout(classifier_dropout), nn.Linear(512, 1),
        )

    def forward(self, spatial_stages: list[Tensor], temporal_stages: list[Tensor]) -> Tensor:
        readouts = [stage(s, t) for stage, s, t in zip(self.stages, spatial_stages, temporal_stages, strict=True)]
        return self.classifier(torch.cat(readouts, dim=-1))


# ---------------------------------------------------------------------------
# The model (thesis principal.py) and its use in the worker
# ---------------------------------------------------------------------------
class SwinFusionPlusPlus(nn.Module):
    """Input: B x 16 x 3 x 224 x 224 RGB, uint8 or float in [0, 1]. Output: B clip logits."""

    def __init__(self) -> None:
        super().__init__()
        self.normalizer = InputNormalizer()
        self.spatial_backbone = SpatialSwinBackbone()
        self.temporal_backbone = TemporalSwinBackbone()
        self.fusion = TLBGHCA(dropout=0.10, classifier_dropout=0.30)

    def taps(self, clip: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        normalized = self.normalizer(clip)
        return self.spatial_backbone(normalized), self.temporal_backbone(normalized)

    def forward(self, clip: Tensor) -> Tensor:
        spatial, temporal = self.taps(clip)
        return self.fusion(spatial, temporal).squeeze(-1)

    def evidence(self, clip: Tensor, autocast=None) -> tuple[Tensor, Tensor]:
        """Clip logits and Grad-CAM maps (B x 8 x 7 x 7, float32) on the stage-4 features of both encoders.

        The backbones run without a graph; only the fusion head is differentiated, from detached copies of the
        stage taps, so the extra memory is that of the small fusion module. Call outside ``torch.inference_mode``.
        """
        context = autocast if autocast is not None else torch.autocast("cpu", enabled=False)
        with torch.no_grad(), context:
            spatial, temporal = self.taps(clip)
        spatial = [t.detach().clone() for t in spatial]
        temporal = [t.detach().clone() for t in temporal]
        spatial[-1].requires_grad_(True)
        temporal[-1].requires_grad_(True)
        with torch.enable_grad(), context:
            logits = self.fusion(spatial, temporal).squeeze(-1)
        grad_s, grad_t = torch.autograd.grad(logits.float().sum(), [spatial[-1], temporal[-1]])

        def cam(activation: Tensor, gradient: Tensor) -> Tensor:  # B,T,C,H,W -> B,T,H,W
            weights = gradient.float().mean(dim=(1, 3, 4), keepdim=True)
            return (weights * activation.detach().float()).sum(dim=2)

        spatial_map = cam(spatial[-1], grad_s)  # 16 frames
        spatial_map = spatial_map.reshape(spatial_map.shape[0], -1, 2, *spatial_map.shape[2:]).mean(dim=2)  # 8 tubelets
        return logits.detach().float(), spatial_map + cam(temporal[-1], grad_t)


def load_swinfusionpp(path: Path, device: torch.device) -> tuple[SwinFusionPlusPlus, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    net = SwinFusionPlusPlus()
    net.load_state_dict(state, strict=True)
    net.requires_grad_(False)
    net.eval().to(device)
    return net, ckpt if isinstance(ckpt, dict) else {}
