"""DFD-FCG deepfake detector: model, face landmarks and face crop.

Han et al., "Towards More General Video-based Deepfake Detection through Facial Component
Guided Adaptation for Foundation Model" (CVPR 2025), https://github.com/aiiu-lab/DFD-FCG.
The authors release their code and weights for research use only. This file is an
inference-only implementation written for this server; it loads the authors' published
checkpoint (trained on FaceForensics++ C23) unchanged.

Model: a frozen CLIP ViT-L/14 image encoder runs on each of the 10 frames of a clip. After
every encoder layer a small adapter reads that layer's attention queries, keys and values:

    spatial   four learned queries (lips, skin, eyes, nose) attend over the 256 patch keys of
              each frame (cosine similarity x 100) and pool the layer's patch embeddings
    temporal  for every patch, how its queries/keys/values relate across the 10 frames
              (a 10 x 10 affinity per head), reduced by two small convolutions to one value
              per patch

The adapters' outputs are summed over the 24 layers and read by three linear heads (spatial,
temporal, both). The clip's fake probability is the mean of the three heads' softmaxes.

Preprocessing follows the authors' scripts (src/preprocess/fetch_landmark_bbox.py and
crop_main_face.py):

    68 landmarks from 2D-FAN (face_alignment 1.4.1), on the frame scaled to at most 800 px
    -> landmarks averaged over +-6 frames, re-centred on the frame's own
    -> similarity transform (8 stable points, LMedS) onto the LRW mean face, 256 x 256
    -> 150 x 150 crop around the mean of landmarks 15..67
    -> 224 x 224, bicubic with antialiasing, CLIP normalisation.

One difference: the authors find faces with S3FD on every frame; here the faces come from the
server's MTCNN detector and identity tracker, and each MTCNN box is converted to the box S3FD
would give (SFD_FROM_MTCNN, measured on sample videos) before 2D-FAN reads it.

Evidence maps: every head is linear in a layer-normalised feature that is itself a sum over
frames and patches, so the sum of the three heads' fake-minus-real logits splits exactly into
one value per frame and patch (see DfdFcg.forward).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

NUM_FRAMES = 10  # frames per clip; fixed by the checkpoint (the temporal adapter works on 10 x 10)
INPUT_SIZE = 224
PATCH = 14
GRID = INPUT_SIZE // PATCH  # 16 x 16 patches per frame
WIDTH, LAYERS, HEADS = 1024, 24, 16
FACE_QUERIES = 4  # lips, skin, eyes, nose
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Face crop (crop_main_face.py defaults)
CANVAS = 256  # the mean face is defined on a 256 x 256 canvas
CROP_SIZE = 150
SMOOTH_WINDOW = 12  # landmarks are averaged over +-6 frames
STABLE_POINTS = (28, 33, 36, 39, 42, 45, 48, 54)  # nose bridge and base, eye corners, mouth corners
CENTRE_FROM = 15  # the crop is centred on the mean of landmarks 15..67

# Landmarks (fetch_landmark_bbox.py and face_alignment 1.4.1)
MAX_LANDMARK_RES = 800  # frames are scaled down to this before the landmark model sees them
FAN_INPUT, FAN_OUTPUT = 256, 64
SFD_REFERENCE_SCALE = 195.0

# The LRW mean face (misc/20words_mean_face.npy in the authors' repository): 68 landmarks.
MEAN_FACE = np.array([
    (70.92383848116445, 97.13757949462641), (72.62515288351078, 114.90361188285254), (75.98941827444136, 131.07402472829844), (79.2695931821559, 146.2116595997759),
    (83.61348894398981, 163.26020701139876), (91.33134185681857, 177.44535288004224), (100.27169244578133, 187.08855670427997), (112.12435016388423, 196.00353535226736),
    (130.7417580075817, 200.5299886205456), (149.85530270092573, 195.3119806538203), (163.10168700103682, 186.44060881414399), (173.65334171534926, 176.5725015818616),
    (182.4361445903047, 162.27992571994974), (187.16738555724973, 145.09391977605495), (190.22905332928843, 129.72418731003313), (193.02118502126157, 113.45923357784717),
    (194.43863372217956, 95.57953920495954), (81.33095966664638, 80.7954451081944), (87.75906555768356, 75.27980275239393), (96.22692544456642, 73.83857496648524),
    (104.55524334700424, 74.74029382053662), (112.23186143963814, 76.97670953698955), (144.49576387205596, 76.42387471498733), (152.34799900707932, 73.83329748466542),
    (161.13054078871582, 72.6357038526189), (170.5871567415716, 73.84785054186868), (178.21409885404546, 79.4380285679228), (128.733742497853, 95.35962565543399),
    (128.48854473271905, 106.92459505614299), (128.24475936187915, 118.27285085803828), (128.2659654694305, 127.69870726985474), (118.7600011311932, 135.19357677297927),
    (122.96307973457223, 136.14619773626936), (128.87017960974802, 137.30253355584708), (134.94283139516824, 135.99720543270277), (139.48259748050492, 134.87763793235783),
    (92.52245552947952, 94.36876013925871), (97.58518219091603, 90.95977780864015), (105.41368272534791, 90.91345887337647), (112.7724172413806, 94.94360869870115),
    (106.1036350029978, 97.0848569305685), (98.0462856522737, 97.36335869283778), (145.52511509158018, 94.5349986248377), (152.5895343844843, 90.21485665622998),
    (160.6117066599242, 90.19938513610091), (166.67710071343456, 93.56562295717796), (160.5597157158167, 96.48125957611816), (152.2046599320386, 96.47281336075321),
    (107.16760614090083, 157.19606764242243), (114.4761121594912, 152.12006956869445), (123.84852759001141, 148.5186319927202), (128.97628287508832, 149.41552526584132),
    (134.14360702818732, 148.4262821063011), (144.17717841903226, 151.7934326195266), (152.19284005377082, 156.98711116207943), (143.85966895195557, 164.0034710111701),
    (136.7441506983738, 167.94300059697187), (129.15278277857666, 168.81853365656912), (121.79511073818607, 168.0227192850946), (115.27508573225096, 164.1515935501153),
    (109.23088653400715, 157.0017210276686), (122.50270761831963, 154.40733648815748), (129.02862235738357, 154.1210422747831), (135.83648069066908, 154.31214997881702),
    (150.7578280913324, 156.79506003838384), (135.66204627207122, 160.62976731508437), (128.95218547222623, 161.28762709004087), (122.48775431575005, 160.50878431332862),
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class _QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class _Attention(nn.Module):
    """CLIP's multi-head self-attention, also returning each head's queries, keys and values."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * width))
        self.out_proj = nn.Linear(width, width)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (frames, tokens, width). q, k, v: (frames, tokens, heads, width / heads)."""
        q, k, v = (a.unflatten(-1, (self.heads, -1)) for a in
                   F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1))
        mixed = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return q, k, v, self.out_proj(mixed.transpose(1, 2).flatten(-2))


class _Block(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = _Attention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, 4 * width)), ("gelu", _QuickGELU()), ("c_proj", nn.Linear(4 * width, width)),
        ]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, out = self.attn(self.ln_1(x))
        x = x + out
        return q, k, v, x + self.mlp(self.ln_2(x))


class _ImageEncoder(nn.Module):
    """The CLIP ViT-L/14 image tower up to its last transformer layer (no pooling, no projection)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, WIDTH, PATCH, PATCH, bias=False)
        self.class_embedding = nn.Parameter(torch.empty(WIDTH))
        self.positional_embedding = nn.Parameter(torch.empty(GRID * GRID + 1, WIDTH))
        self.ln_pre = nn.LayerNorm(WIDTH)
        self.blocks = nn.ModuleList(_Block(WIDTH, HEADS) for _ in range(LAYERS))
        self.ln_post = nn.LayerNorm(WIDTH)  # in the checkpoint, but the detector never pools the class token

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.conv1(frames).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        return self.ln_pre(torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype))


class _Adapter(nn.Module):
    """The side decoder attached to one encoder layer ("SynoBlock" in the authors' code)."""

    def __init__(self) -> None:
        super().__init__()
        self.t_conv = nn.Conv2d(3 * HEADS, 1, 5, padding=2)
        self.t_proj = nn.Sequential(nn.LayerNorm(NUM_FRAMES**2), nn.Linear(NUM_FRAMES**2, NUM_FRAMES), nn.GELU(),
                                    nn.Linear(NUM_FRAMES, NUM_FRAMES**2))
        self.p_conv = nn.Conv2d(NUM_FRAMES**2, 1, 5, padding=2)
        self.syno_embedding = nn.Parameter(torch.zeros(FACE_QUERIES, WIDTH))

    def temporal(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, clips: int) -> torch.Tensor:
        """q, k, v: patch tokens (frames, 256, heads, dim), float32. Returns (clips, 256)."""
        affinities = []
        for attr in (q, k, v):
            a = attr.unflatten(0, (clips, NUM_FRAMES)).permute(0, 2, 3, 1, 4)  # clips, patches, heads, frames, dim
            affinity = (a / (a.size(-1) ** 0.5)) @ a.transpose(-1, -2)  # frame x frame, per patch and head
            affinities.append(affinity.softmax(dim=-1).flatten(0, 1))
        x = self.t_conv(torch.cat(affinities, dim=1))  # (clips * patches, 1, frames, frames)
        x = x.unflatten(0, (clips, GRID, GRID)).flatten(3)
        x = x + self.t_proj(x)
        return self.p_conv(x.permute(0, 3, 1, 2)).flatten(1)

    def spatial(self, k: torch.Tensor, emb: torch.Tensor, clips: int) -> tuple[torch.Tensor, torch.Tensor]:
        """k: patch keys (frames, 256, heads, dim); emb: patch embeddings (frames, 256, width); float32.

        Returns the pooled feature (clips, width) and the queries' attention (frames, 4, 256)."""
        keys = k.flatten(-2)
        query = self.syno_embedding
        attention = (query / (query.norm(dim=-1, keepdim=True) + 1e-4)) @ \
            (keys / (keys.norm(dim=-1, keepdim=True) + 1e-4)).transpose(-1, -2)
        attention = (attention * 100).softmax(dim=-1)
        pooled = (attention @ emb).unflatten(0, (clips, NUM_FRAMES)).flatten(1, 2).mean(dim=1)
        return pooled, attention


class DfdFcg(nn.Module):
    """Input: (clips, 10, 3, 224, 224), CLIP-normalised RGB. See ``forward`` for the output."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _ImageEncoder()
        self.adapters = nn.ModuleList(_Adapter() for _ in range(LAYERS))
        self.s_ln = nn.LayerNorm(WIDTH)
        self.s_head = nn.Linear(WIDTH, 2)
        self.t_ln = nn.LayerNorm(GRID * GRID)
        self.t_head = nn.Linear(GRID * GRID, 2)
        self.a_head = nn.Linear(WIDTH + GRID * GRID, 2)
        # Evidence-map read-out, derived from the heads in ``_prepare_evidence``.
        self.register_buffer("spatial_evidence", torch.zeros(WIDTH), persistent=False)
        self.register_buffer("temporal_evidence", torch.zeros(GRID * GRID), persistent=False)
        self.register_buffer("evidence_bias", torch.zeros(()), persistent=False)

    @torch.no_grad()
    def _prepare_evidence(self) -> None:
        """Each head's fake-minus-real logit is ``w . LayerNorm(y) + b``. With u = w * gamma and
        sigma the layer norm's standard deviation, that is ``(u - mean(u)) . y / sigma`` plus a
        constant, which is linear in y. The spatial feature y is a sum over frames and patches and
        the temporal feature has one entry per patch, so the logit splits over frames and patches."""
        def margin(weight: torch.Tensor) -> torch.Tensor:
            return (weight[1] - weight[0]).float()

        def centred(u: torch.Tensor) -> torch.Tensor:
            return u - u.mean()

        ws, wt, wa = margin(self.s_head.weight), margin(self.t_head.weight), margin(self.a_head.weight)
        was, wat = wa[:WIDTH], wa[WIDTH:]
        gs, gt = self.s_ln.weight.float(), self.t_ln.weight.float()
        self.spatial_evidence.copy_(centred(ws * gs) + centred(was * gs))
        self.temporal_evidence.copy_(centred(wt * gt) + centred(wat * gt))
        bias = (ws + was) @ self.s_ln.bias.float() + (wt + wat) @ self.t_ln.bias.float()
        bias = bias + margin(self.s_head.bias) + margin(self.t_head.bias) + margin(self.a_head.bias)
        self.evidence_bias.copy_(bias)

    def forward(self, clips: torch.Tensor, evidence: bool = False) -> dict[str, torch.Tensor]:
        """Returns, per clip:

        prob_fake  the authors' fused score: mean of the three heads' softmaxes (+1e-4), renormalised
        logit      log-odds of that score
        margins    the spatial, temporal and joint heads' fake-minus-real logits, (clips, 3)
        evidence   (clips, 10, 16, 16), only if asked for: each frame patch's share of
                   ``margins.sum(1)``; ``evidence.sum((1, 2, 3)) + evidence_bias`` equals it
        """
        count = clips.shape[0]
        x = self.backbone.embed(clips.flatten(0, 1))
        syno_s: torch.Tensor | float = 0.0
        syno_t: torch.Tensor | float = 0.0
        heat = None
        for block, adapter in zip(self.backbone.blocks, self.adapters):
            q, k, v, x = block(x)
            # The adapters run in float32: a softmax over cosine similarity x 100 magnifies
            # half-precision rounding. They cost next to nothing beside the encoder.
            with torch.autocast(x.device.type, enabled=False):
                q, k, v, emb = q[:, 1:].float(), k[:, 1:].float(), v[:, 1:].float(), x[:, 1:].float()
                pooled, attention = adapter.spatial(k, emb, count)
                syno_s = syno_s + pooled
                syno_t = syno_t + adapter.temporal(q, k, v, count)
                if evidence:
                    layer = attention.sum(dim=1) * (emb @ self.spatial_evidence)  # (frames, 256)
                    heat = layer if heat is None else heat + layer
        with torch.autocast(x.device.type, enabled=False):
            feature_s, feature_t = self.s_ln(syno_s), self.t_ln(syno_t)
            logits = torch.stack([self.s_head(feature_s), self.t_head(feature_t),
                                  self.a_head(torch.cat([feature_s, feature_t], dim=-1))], dim=1)  # (clips, 3, 2)
            fused = logits.softmax(dim=-1).mean(dim=1) + 1e-4
            out = {
                "prob_fake": fused[:, 1] / fused.sum(dim=-1),
                "logit": fused[:, 1].log() - fused[:, 0].log(),
                "margins": logits[..., 1] - logits[..., 0],
            }
            if evidence:
                eps = self.s_ln.eps
                sigma_s = (syno_s.var(dim=-1, unbiased=False) + eps).sqrt()
                sigma_t = (syno_t.var(dim=-1, unbiased=False) + eps).sqrt()
                spatial = heat.unflatten(0, (count, NUM_FRAMES)) / (NUM_FRAMES * FACE_QUERIES * sigma_s[:, None, None])
                temporal = self.temporal_evidence * syno_t / sigma_t[:, None]
                out["evidence"] = (spatial + temporal[:, None] / NUM_FRAMES).unflatten(-1, (GRID, GRID))
        return out


def _checkpoint_key(key: str) -> str | None:
    """The authors' state-dict key -> this module's."""
    if not key.startswith("model."):
        return None
    key = key[len("model."):]
    for old, new in (("encoder.model.transformer.resblocks.", "backbone.blocks."), ("encoder.model.", "backbone."),
                     ("encoder.decoder.decoder_layers.", "adapters.")):
        if key.startswith(old):
            return new + key[len(old):]
    return key


def load_dfd_fcg(path: Path, device: torch.device) -> tuple[DfdFcg, dict]:
    """Load the authors' Lightning checkpoint (weights.ckpt). Returns the model and the checkpoint's
    scalar details (epoch, ...)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = {}
    for key, value in ckpt["state_dict"].items():
        name = _checkpoint_key(key)
        if name is None:
            raise ValueError(f"Unexpected key in the DFD-FCG checkpoint: {key}")
        state[name] = value
    net = DfdFcg()
    net.load_state_dict(state, strict=True)
    net._prepare_evidence()
    net.requires_grad_(False)
    net.eval().to(device)
    return net, {"epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step")}


def prepare_frames(crops: torch.Tensor) -> torch.Tensor:
    """(N, 150, 150, 3) uint8 face crops -> (N, 3, 224, 224) model input.

    The authors' transform: torchvision ``Resize(224, BICUBIC, antialias=True)`` on the uint8
    frames (which rounds back to uint8), then /255 and CLIP normalisation."""
    x = crops.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bicubic", align_corners=False, antialias=True)
    x = x.clamp_(0, 255).round_().div_(255.0)
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Landmarks: 2D-FAN, as face_alignment 1.4.1 runs it
# ---------------------------------------------------------------------------
# S3FD box from an MTCNN box, both (x1, y1, x2, y2): centre shift and size as fractions of the
# MTCNN box's width and height. Medians over 752 faces in 11 sample videos (FF++, Celeb-DF v2,
# DeeperForensics; 480p to 1080p) where both detectors ran; the square 2D-FAN then looks at lands
# within 1% (centre) and 3.5% (size) of the one the real S3FD box gives, see README "Vidi: DFD-FCG".
SFD_FROM_MTCNN = {"shift_x": 0.0128, "shift_y": -0.0049, "width": 0.9894, "height": 0.9931}


def sfd_boxes(boxes: np.ndarray) -> np.ndarray:
    """MTCNN boxes (K, 4) -> the boxes S3FD gives for the same faces."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    width, height = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 + SFD_FROM_MTCNN["shift_x"] * width
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 + SFD_FROM_MTCNN["shift_y"] * height
    half_w, half_h = SFD_FROM_MTCNN["width"] * width / 2, SFD_FROM_MTCNN["height"] * height / 2
    return np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)


def landmark_weights(weights_dir: Path, name: str) -> Path:
    """Where the 2D-FAN TorchScript file is: ORIGINAI_FAN_WEIGHTS, next to the checkpoints, or in
    the image's torch cache (the Dockerfile downloads it there)."""
    override = os.environ.get("ORIGINAI_FAN_WEIGHTS", "").strip()
    places = [Path(override)] if override else []
    places += [Path(weights_dir) / name, Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "checkpoints" / name]
    for place in places:
        if place.is_file():
            return place
    raise FileNotFoundError(f"The landmark model {name} was not found in: {', '.join(map(str, places))}")


class Landmarker:
    """68 landmarks per face from 2D-FAN, computed the way the authors' script does:
    ``FaceAlignment(TWO_D, flip_input=False).get_landmarks_from_batch`` on frames scaled to at
    most 800 px, half precision on the GPU."""

    def __init__(self, weights: Path, device: torch.device) -> None:
        self.device = device
        self.net = torch.jit.load(str(weights), map_location="cpu").eval()
        self.half = device.type == "cuda"
        self.net.to(device, dtype=torch.float16 if self.half else torch.float32)

    CHUNK = 16  # faces per network call; always this many (see ``run``)

    @torch.inference_mode()
    def __call__(self, frames: torch.Tensor, frame_inds: list[int], boxes: np.ndarray) -> np.ndarray:
        """frames: (B, 3, H, W) uint8 on the device. ``boxes`` (K, 4): one S3FD-style box per face,
        in frame pixels, for the face in ``frames[frame_inds[k]]``.

        Returns (K, 68, 2) float32 in frame pixels: whole pixels of the scaled frame, divided by
        the scale (as the authors store them). Faces whose box is unusable get NaN."""
        inputs, geometry = self.inputs(frames, frame_inds, boxes)
        return self.points(self.run(inputs), geometry)

    @torch.inference_mode()
    def inputs(self, frames: torch.Tensor, frame_inds: list[int], boxes: np.ndarray) -> tuple[torch.Tensor, dict]:
        """The network inputs for these faces, (k, 256, 256, 3) uint8 on the device (one per usable
        box), and what ``points`` needs to turn the network's answer into frame pixels.

        Splitting this from ``run`` lets the caller collect faces from several frame batches and
        run the network on full chunks only."""
        count = len(frame_inds)
        geometry = {"count": count, "keep": np.zeros(0, dtype=np.int64)}
        empty = torch.empty((0, FAN_INPUT, FAN_INPUT, 3), dtype=torch.uint8, device=self.device)
        if not count:
            return empty, geometry
        height, width = frames.shape[2:]
        scale = MAX_LANDMARK_RES / max(height, width) if max(height, width) > MAX_LANDMARK_RES else 1
        if scale != 1:  # cv2.resize(frame, None, fx=scale, fy=scale): bilinear, half-pixel centres
            hs, ws = round(height * scale), round(width * scale)
            scaled = F.interpolate(frames.float(), size=(hs, ws), mode="bilinear", align_corners=False).round_()
        else:
            hs, ws, scaled = height, width, frames
        flat = scaled.permute(0, 2, 3, 1).reshape(len(frames), hs * ws, 3)

        # face_alignment: the box centre moved up by 12% of its height, and the size of the square
        # the landmark model looks at (200 x scale pixels).
        d = (np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * scale).astype(np.float32)
        cx = d[:, 2] - (d[:, 2] - d[:, 0]) / np.float32(2.0)
        cy = d[:, 3] - (d[:, 3] - d[:, 1]) / np.float32(2.0)
        cy = cy - (d[:, 3] - d[:, 1]) * np.float32(0.12)
        side = 200.0 * ((d[:, 2] - d[:, 0] + d[:, 3] - d[:, 1]).astype(np.float64) / SFD_REFERENCE_SCALE)
        # utils.crop: the square from transform((1, 1)) to transform((256, 256)), truncated to pixels
        side32 = side.astype(np.float32)
        ul = np.stack([cx - side32 / 2 + side32 / FAN_INPUT, cy - side32 / 2 + side32 / FAN_INPUT], axis=1).astype(np.int64)
        br = np.stack([cx + side32 / 2, cy + side32 / 2], axis=1).astype(np.int64)
        usable = np.isfinite(side) & ((br - ul) >= 2).all(axis=1)
        keep = np.flatnonzero(usable)
        if not len(keep):
            return empty, geometry

        inds = torch.as_tensor(np.asarray(frame_inds)[keep], device=self.device)
        ul_t, br_t = torch.as_tensor(ul[keep], device=self.device), torch.as_tensor(br[keep], device=self.device)
        # utils.crop rounds to whole values in 0..255, so uint8 holds the inputs exactly.
        crops = torch.cat([self._crops(flat, hs, ws, inds[s : s + self.CHUNK], ul_t[s : s + self.CHUNK], br_t[s : s + self.CHUNK])
                           for s in range(0, len(keep), self.CHUNK)]).to(torch.uint8)
        geometry.update(keep=keep, side=side[keep], cx=cx[keep], cy=cy[keep], scale=scale)
        return crops, geometry

    @torch.inference_mode()
    def run(self, crops: torch.Tensor) -> torch.Tensor:
        """The network on (k, 256, 256, 3) uint8 inputs -> (k, 68, 2) heatmap peaks on the 64 x 64
        grid, on the device. Always ``CHUNK`` faces per call: TorchScript re-optimises the network
        for every new input shape, which made this step take 1 to 30 seconds per video. Each face
        is independent (eval mode), so the zero padding of a last, partial chunk changes nothing."""
        heatmaps = []
        for s in range(0, len(crops), self.CHUNK):
            inp = crops[s : s + self.CHUNK].permute(0, 3, 1, 2).float().div(255.0)
            count = len(inp)
            if count < self.CHUNK:
                inp = torch.cat([inp, inp.new_zeros((self.CHUNK - count, *inp.shape[1:]))])
            heatmaps.append(self.net(inp.half() if self.half else inp).float()[:count])
        if not heatmaps:
            return torch.empty((0, 68, 2), device=self.device)
        return self._peaks(torch.cat(heatmaps))

    @staticmethod
    def points(peaks: torch.Tensor | np.ndarray, geometry: dict) -> np.ndarray:
        """(count, 68, 2) float32 frame pixels from ``run``'s peaks for the faces ``inputs`` kept."""
        points = np.full((geometry["count"], 68, 2), np.nan, dtype=np.float32)
        keep = geometry["keep"]
        if not len(keep):
            return points
        peaks = peaks.double().cpu().numpy() if isinstance(peaks, torch.Tensor) else np.asarray(peaks, dtype=np.float64)
        # utils.transform_np(invert=True) from the 64 x 64 grid to the scaled frame, truncated to
        # whole pixels, then back to the frame's own resolution.
        side, cx, cy = geometry["side"], geometry["cx"], geometry["cy"]
        cell = side / FAN_OUTPUT
        origin = np.stack([cx, cy], axis=1).astype(np.float64) - side[:, None] / 2
        pixels = np.trunc(peaks * cell[:, None, None] + origin[:, None, :])
        points[keep] = pixels.astype(np.float32) / np.float32(geometry["scale"])
        return points

    def _crops(self, flat: torch.Tensor, hs: int, ws: int, inds: torch.Tensor, ul: torch.Tensor,
               br: torch.Tensor) -> torch.Tensor:
        """utils.crop for a batch: the square [ul, br) of each frame (black outside the frame),
        resized to 256 x 256 as cv2.resize(INTER_LINEAR) does. Returns (k, 256, 256, 3) float."""
        size = FAN_INPUT
        centres = torch.arange(size, device=flat.device, dtype=torch.float64) + 0.5
        span = (br - ul).double()  # (k, 2): width, height of the square

        def taps(axis: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            n = span[:, axis : axis + 1]
            pos = centres[None] * (n / size) - 0.5
            first = pos.floor()
            weight = pos - first
            low, high = first < 0, first >= n - 1
            first = torch.where(low, torch.zeros_like(first), torch.where(high, n - 1, first))
            weight = torch.where(low | high, torch.zeros_like(weight), weight)
            second = torch.minimum(first + 1, n - 1)
            offset = ul[:, axis : axis + 1]
            return first.long() + offset, second.long() + offset, weight.float()

        x0, x1, wx = taps(0)
        y0, y1, wy = taps(1)
        b = inds.long()[:, None, None]

        def px(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            y, x = y[:, :, None], x[:, None, :]
            inside = (x >= 0) & (x < ws) & (y >= 0) & (y < hs)
            values = flat[b, y.clamp(0, hs - 1) * ws + x.clamp(0, ws - 1)].float()
            return values * inside[..., None]

        wx, wy = wx[:, None, :, None], wy[:, :, None, None]
        top = px(y0, x0) * (1 - wx) + px(y0, x1) * wx
        bottom = px(y1, x0) * (1 - wx) + px(y1, x1) * wx
        return (top * (1 - wy) + bottom * wy).round_()

    @staticmethod
    def _peaks(heatmaps: torch.Tensor) -> torch.Tensor:
        """utils.get_preds_fromhm: each heatmap's maximum, moved a quarter cell toward its higher
        neighbour, as a position on the 64 x 64 grid (x, y)."""
        n = FAN_OUTPUT
        flat = heatmaps.flatten(2)
        index = flat.argmax(dim=-1)
        px, py = index % n, index // n

        def at(dy: int, dx: int) -> torch.Tensor:
            return flat.gather(-1, ((py + dy).clamp(0, n - 1) * n + (px + dx).clamp(0, n - 1))[..., None])[..., 0]

        inner = (px > 0) & (px < n - 1) & (py > 0) & (py < n - 1)
        dx = torch.where(inner, torch.sign(at(0, 1) - at(0, -1)) * 0.25, torch.zeros_like(flat[..., 0]))
        dy = torch.where(inner, torch.sign(at(1, 0) - at(-1, 0)) * 0.25, torch.zeros_like(flat[..., 0]))
        return torch.stack([px + 1 + dx - 0.5, py + 1 + dy - 0.5], dim=-1)


# ---------------------------------------------------------------------------
# Face crop geometry (crop_main_face.py: crop_patch, affine_transform, crop_driver)
# ---------------------------------------------------------------------------
def crop_geometry(points: np.ndarray, frame_index: np.ndarray, total_frames: int) -> list[tuple[np.ndarray, int, int] | None]:
    """Where each tracked frame's 150 x 150 face crop comes from.

    ``points`` (N, 68, 2) float32: the tracked face's landmarks, in frame order; ``frame_index``
    (N,): each one's frame number in the video; ``total_frames``: the video's frame count.

    Returns per frame ``(matrix, left, top)``: the 2 x 3 transform from the frame onto the
    256 x 256 mean-face canvas and the crop's top-left corner on that canvas, or None if the
    transform could not be estimated.
    """
    points = np.asarray(points, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)
    return [frame_crop_geometry(points, frame_index, i, total_frames) for i in range(len(frame_index))]


_REFERENCE = MEAN_FACE[list(STABLE_POINTS)].copy()


def frame_crop_geometry(points: np.ndarray, frame_index: np.ndarray, i: int,
                        total_frames: int) -> tuple[np.ndarray, int, int] | None:
    """``crop_geometry`` for the i-th frame. It reads only the frames within 6 of it, so it can be
    called as soon as those are known (``points`` (N, 68, 2) float32 and ``frame_index`` (N,) int64
    in frame order, holding at least every tracked frame up to 6 after this one)."""
    half = CROP_SIZE // 2
    frame = frame_index[i]
    margin = min(SMOOTH_WINDOW // 2, int(frame), total_frames - 1 - int(frame))
    lo = np.searchsorted(frame_index, frame - margin, side="left")
    hi = np.searchsorted(frame_index, frame + margin, side="right")
    smoothed = np.mean(points[lo:hi], axis=0)
    smoothed += points[i].mean(axis=0) - smoothed.mean(axis=0)
    matrix = cv2.estimateAffinePartial2D(smoothed[list(STABLE_POINTS)], _REFERENCE, method=cv2.LMEDS)[0]
    if matrix is None or not np.isfinite(matrix).all():
        return None
    moved = np.matmul(smoothed, matrix[:, :2].transpose()) + matrix[:, 2].transpose()
    cx, cy = np.mean(moved[CENTRE_FROM:], axis=0)
    if cy - half < 0:
        cy = half + 1
    elif cy + half > CANVAS:
        cy = CANVAS - half - 1
    if cx - half < 0:
        cx = half + 1
    elif cx + half > CANVAS:
        cx = CANVAS - half - 1
    return matrix, int(cx - half), int(cy - half)


def input_matrix(matrix: np.ndarray, left: int, top: int) -> np.ndarray:
    """Frame -> 224 x 224 model input (the 150 x 150 crop resized with half-pixel centres)."""
    scale = INPUT_SIZE / CROP_SIZE
    out = np.asarray(matrix, dtype=np.float64) * scale
    out[0, 2] = (matrix[0, 2] - left + 0.5) * scale - 0.5
    out[1, 2] = (matrix[1, 2] - top + 0.5) * scale - 0.5
    return out


def protocol_frames(total_frames: int, fps: float, max_clips: int = 100) -> list[list[int]]:
    """Frame numbers of the clips the authors' evaluation reads from a video (src/dataset):
    one clip per whole 3 seconds, 10 frames spread evenly over it.

    Their face-crop file keeps every frame at round(fps), and the reader seeks to the frame
    nearest each sample time."""
    rate = int(round(fps))
    if rate <= 0:
        return []
    clips = min(int((total_frames / rate) // 3), max_clips)
    stride = (int(rate * 3) - 1) / (NUM_FRAMES - 1)
    return [[int(round(clip * 3 * rate + sample * stride)) for sample in range(NUM_FRAMES)] for clip in range(clips)]
