# OriginAI on Vast.ai Serverless

Serves four detectors (their checkpoints are kept locally in the project's `models/` folder) from one Vast.ai
Serverless endpoint: three of the thesis's Video Swin-B models, and DFD-FCG as "Vidi" (see
[Vidi: DFD-FCG](#vidi-dfd-fcg)). Vast does not turn a model into an API by itself: `model_server/` *is*
the API. Vast runs it on GPUs on demand and routes signed requests to it. The website's
`backend/server.py` is the only thing that talks to Vast, because every Vast call needs the
Vast API key.

```
Browser ──multipart──► website server.py ──JSON (base64 video)──► Vast route ──► GPU worker
        ◄── polls /api/analyses/{id}; loads evidence images from /api/analyses/{id}/media/…
                                                              PyWorker :3000 ─► model server :18000
```

| Model id  | Name    | Model        | Checkpoint             | Temperature T | Threshold* | Frame spacing Δ |
|-----------|---------|--------------|------------------------|---------------|------------|-----------------|
| `veni-hq` | Veni HQ | Video Swin-B | FF++ C23.pt            | 1.992         | 0.525      | 0.340 s |
| `veni-lq` | Veni LQ | Video Swin-B | FF++ C40.pt            | 2.559         | 0.489      | 0.340 s |
| `vidi`    | Vidi    | DFD-FCG      | DFD-FCG.ckpt           | 1 (none)      | 0.5        | 0.33 s  |
| `vici`    | Vici    | Video Swin-B | DeeperForensics-1.0.pt | 1.396         | 0.832      | 0.345 s |

\* For the Video Swin-B models the threshold applies to the calibrated probability
`sigmoid(mean_logit / T)`, exactly as in `Thesis_Code/src/swinfusionpp/p11_metric_stage.py`, and Δ is the
median per-video frame spacing of each model's training split
(`manifests/preprocessing_materialization_v001.parquet`). DFD-FCG has no validation run of ours: it is read at
its authors' decision rule (mean clip probability ≥ 0.5).

Until October 2026 Vidi was the thesis's Video Swin-B model trained on Celeb-DF v2 (`Celeb-DF v2.pt`, T 5.016,
threshold 0.229, Δ 0.245 s). That checkpoint is still in `models/` and in the Backblaze bucket; putting its
entry back in `models.json` restores it.

Everything from here to [Vidi: DFD-FCG](#vidi-dfd-fcg) describes the Video Swin-B models. The two kinds of
model share decoding, face detection, tracking, the timeline and the rendering of evidence maps.

## Preprocessing: the frozen thesis contract, on the GPU

Ported from `Thesis_Code/src/swinfusionpp/preprocessing.py` and
`configs/preprocessing/policy_v001.yaml` (policy `preprocessing_v001`), and stored in
`model_server/models.json → policy`. The heavy steps run on the GPU and were verified against
the original CPU code:

| Step | Implementation | Check against training |
|---|---|---|
| Decode | torchcodec: NVDEC on the GPU. Falls back to FFmpeg or PyAV on the CPU | CPU decode is bit-identical to the thesis PyAV rgb24 frames (40/40) |
| Face detection | `gpu_mtcnn.py`: facenet-pytorch 2.6.0 MTCNN, same weights and thresholds. Stages 2 and 3 crop every box in one gather using an exact summed-area table instead of a Python loop | Identical boxes, probabilities and landmarks (max difference 0 at 640×480 and 1080p) |
| Alignment | Fixed-point bilinear warp on the GPU to the ArcFace template, 224×224, REFLECT_101 | Pixel-identical to OpenCV 4.11 `warpAffine` (60/60 crops) |
| Identity tracking | FaceNet vggface2 embeddings on the GPU; the thesis tracker logic runs on the CPU (tiny) | Same rule and thresholds |

The model is torchvision `swin3d_b` on 16 frames (B×16×3×224×224) with a `Linear(1024 → 1)` head, run in bf16 on the GPU as in training.

**Verified on `000_003.mp4`** (FF++ C23 test split): the thesis 3-clip logits are 11.785 / 11.944 / 12.271, against 11.63 / 11.94 / 12.31 recorded in training (bf16).

## Every frame is analysed

The model only accepts 16 frames at the spacing it was trained on (0.245–0.345 s apart), so
the video is split into **interleaved phases**:
- **Phases:** phase *p* holds frames *p*, *p+k*, *p+2k*, … where *k* = round(spacing × fps). For example, 25 fps with 0.34 s spacing gives k = 8.
- **Clips:** each phase is tiled into 16-frame clips, and each phase starts at a different offset so clip centres spread evenly through the video. Every face frame is scored in at least one clip, and always at the trained spacing.
- **Verdict:** mean of all clip logits → `sigmoid(mean / T)` → threshold, the thesis read-out.
- **Timeline:** each bar averages the clips centred in a half-clip window.
- **Benchmark protocol:** the exact thesis evaluation (64 timestamps → 48 tracked frames → three anchors) is reported next to it.
- **Limits:** videos over `max_frames` (7,200) are subsampled evenly. Clips never bridge a face gap longer than 4× the spacing.

Portrait phone videos are rotated upright before face detection.

## Evidence maps ("where the model looked")

**How they're computed:**
- The Video Swin-B head is LayerNorm → average pool → linear.
- So each final-stage token (8×7×7 per clip) contributes exactly `w · LayerNorm(token)` to the clip logit, and the mean of those contributions plus the bias is the logit.
- This equals Grad-CAM at the final norm layer and comes free with scoring.

**What's rendered:**
- Maps are made for the 4 most suspicious clips spread over the video, plus the least suspicious, and scaled relative within the video.
- Each clip shows 8 frames (one per 2-frame tubelet).
- Rendering happens on the GPU: the heat is blended into the video frame and the face crop and JPEG-encoded with nvJPEG. The page's opacity slider mixes the plain and blended images, which is equivalent to scaling the overlay.

**Limits:**
- The resolution is 7×7 per tubelet, so maps show regions, not pixels.
- Brightness is relative, so it's a guide, not a measurement.

## Worker API (`model_server/app.py`)

`POST /analyze` takes `{"model": id, "video_b64" | "video_url", "filename"}`.

**Success** returns:
- `status: "completed"`, `pred`, `prob_fake`, `raw_prob_fake`, `threshold`, `temperature`, `video_logit`, `aggregation`
  (`mean_clip_logit` for Video Swin, `mean_clip_probability` for DFD-FCG)
- `clips[{index, start_s, end_s, logit, prob, clips, heatmap}]` (timeline windows)
- `protocol{available, pred, prob_fake, clip_logits | clip_probs, clip_count | reason, message}`
- `coverage{mode, decoder, device, frames_decoded, frames_analyzed, face_frames, phases, frame_spacing_s, clips, analyzed_start_s, analyzed_end_s, fallback, …}`
- `heatmaps{target, frame_size, stops, clips[{clip_index, role, clip_start_s, clip_end_s, clip_prob, frames[{t, frame_jpg, frame_heat_jpg, crop_jpg, crop_heat_jpg}]}]}` (base64 JPEGs, about 2 MB)
- `timings`

**Failure** returns `{"status": "failed", "reason", "error"}`, where `reason` is one of:
`no_face`, `low_detection_confidence`, `identity_track_failed`, `insufficient_valid_frames`, `decode_failed`, `metadata_error`.

**Proxy.** The website proxy (`backend/server.py`) exposes `POST /api/analyses`, `GET /api/analyses/{id}` and `GET /api/analyses/{id}/media/{file}`. It writes the heatmap images to `backend/outputs/analyses/<id>/`, replaces the base64 with URLs, and deletes them after an hour.

## Vidi: DFD-FCG

Since October 2026, Vidi is DFD-FCG: Han, Huang, Hua and Chen, *Towards More General Video-based Deepfake
Detection through Facial Component Guided Adaptation for Foundation Model* (CVPR 2025,
[code](https://github.com/aiiu-lab/DFD-FCG), [paper](https://arxiv.org/abs/2404.05583)). The worker serves the
authors' published checkpoint, trained on FaceForensics++ C23, unchanged (`weights.ckpt` from their
`checkpoint.zip`; SHA-256 `70a47c827ede1ddca716404ec518913e26eb316ab57e41528fb191aaae7e4db6`, stored in the bucket as
`OriginAI/models/dfd-fcg/weights.ckpt`). **The authors release their code and weights for research use only**;
anything else needs their permission (readme of their repository).

**Model** (`model_server/dfd_fcg.py`, an inference-only implementation written for this server; it loads the
authors' checkpoint with every weight accounted for):

- A frozen CLIP ViT-L/14 image encoder runs on each of the 10 frames of a clip.
- After each of its 24 layers a small adapter reads that layer's attention queries, keys and values: four
  learned facial-component queries (lips, skin, eyes, nose) attend over the frame's 256 patches, and a temporal
  branch looks at how each patch's features relate across the 10 frames (a 10 x 10 affinity per head, reduced by
  two small convolutions to one value per patch).
- Three linear heads (spatial, temporal, both) read the adapters' outputs summed over the layers. A clip's fake
  probability is the mean of the three heads' softmaxes; the video's score is the mean over its clips, and the
  verdict is DEEPFAKE at 0.5 (the authors' rule; no calibration of ours, so `temperature` is 1).
- On the GPU the encoder runs in float16 under autocast, as in the authors' evaluation (`precision: 16`); the
  adapters and heads run in float32, because a softmax over cosine similarity x 100 magnifies half-precision
  rounding.

**Preprocessing** follows the authors' scripts (`src/preprocess/fetch_landmark_bbox.py`, `crop_main_face.py`):

| Step | The authors | This worker |
|---|---|---|
| Faces | S3FD on every frame, then their own landmark-motion tracker picks the longest-lived face | The MTCNN detector and FaceNet identity tracker shared with the other detectors. Each MTCNN box is converted to the box S3FD gives for the same face (`SFD_FROM_MTCNN`: medians over 752 faces of 11 sample videos; the square 2D-FAN then looks at lands within 1% of its side and 3.5% of its size of the real S3FD one) |
| Landmarks | 2D-FAN (face_alignment 1.4.1), frame scaled to at most 800 px, half precision | The same TorchScript model and the same steps (`dfd_fcg.Landmarker`), batched on the GPU |
| Alignment | Landmarks averaged over +-6 frames and re-centred; similarity transform of 8 stable points onto the LRW mean face, 256 x 256 (`cv2.estimateAffinePartial2D`, LMedS) | The same code path (`crop_geometry`) |
| Crop | 150 x 150 around the mean of landmarks 15..67, `cv2.warpAffine`, black border | The same, cut from the full-resolution frame in a second decoding pass (`FacePipeline.canvas_crops`): the crop needs landmarks of the frames around it and the tracker's choice of face, which are only known after the first pass |
| Input | 224 x 224 bicubic with antialiasing, CLIP normalisation | The same (`prepare_frames`) |

**Checked against the authors' code** (their preprocessing and model classes, run unmodified on the CPU, on the
two videos in their repository and this project's sample clips):

- Model: a clip gets the same probability from both implementations to 7 decimal places.
- Crop code: given the authors' landmarks, every crop is pixel-identical to theirs (1,619 of 1,619 frames in
  4 videos), and the GPU warp equals `cv2.warpAffine` for faces that leave the frame.
- Whole pipeline: landmarks differ by 0.7 to 1.1 px on average (the face boxes differ slightly), which moves a
  crop by up to a pixel in about a tenth of the frames. Under the authors' evaluation protocol the video scores
  were 0.371 vs 0.374 (their `000.mp4`, real), 0.928 vs 0.934 (`000_003.mp4`, fake; their demo shows 0.94),
  0.661 vs 0.651 (Celeb-DF `id0_id9_0005`) and 0.490 vs 0.474 (DeeperForensics `054_M117`).

**Every frame.** As for the other detectors, the tracked frames are split into k = round(0.33 x fps) interleaved
phases and each phase is tiled into 10-frame clips (the trained spacing: 10 frames over 3 seconds), so every
frame is scored once at that spacing. The second check (`protocol`) is the authors' own evaluation: one clip per
whole 3 seconds of video, its 10 frames spread evenly, the video scored by the mean clip probability (at most 100
clips). A clip whose frames are not all tracked is left out.

**Evidence maps.** Each head is linear in a layer-normalised feature; the spatial feature is a sum over frames and
patches of attention-weighted patch embeddings, and the temporal feature has one entry per patch. So the sum of
the three heads' fake-minus-real logits splits exactly into one value per frame and 16 x 16 patch
(`DfdFcg.forward(evidence=True)`; the temporal part is shared equally by the clip's frames). The maps are shown
like the Video Swin ones: for the most suspicious clips and the least, relative within the video, on the 224 x 224
input (the 150 x 150 crop) and on the frame, 10 frames per clip.

**Cost.** The landmark model runs on every face, and the video is decoded twice. The clips of a 2-minute video
are about 3,600 frames through a ViT-L/14.

## Deploy

Repository layout (this repo is public and holds code only):

- `worker.py`, `requirements.txt`: the Vast PyWorker. The template's `PYWORKER_REPO` points at this repo.
- `model_server/`: the model API (`app.py`), the face pipeline and Video Swin detector (`detector.py`), DFD-FCG (`dfd_fcg.py`), registry (`models.json`) and weight downloader (`fetch_weights.py`).
- `Dockerfile`, `onstart.sh`: the worker image, built by `.github/workflows/build-image.yml` into `ghcr.io/<owner>/originai-worker`. The image also holds the public FaceNet and 2D-FAN weights, checked by hash.

The checkpoints stay private in the Backblaze bucket. Each worker downloads them at start with a read-only key (`B2_KEY_ID`, `B2_APP_KEY` in the Vast template) and checks every file's SHA-1 against `models.json`.

1. **Image:** pushing to `main` builds `:latest` / `:cu126`. For RTX 50-series hosts, run the workflow manually with `cu130`.
2. **Vast template:**
   - image: `ghcr.io/<owner>/originai-worker:cu126`
   - Docker options: `-p 3000:3000 -e PYWORKER_REPO=<this repo> -e B2_KEY_ID=… -e B2_APP_KEY=…`
   - on-start: `bash /onstart.sh`
3. **Endpoint settings:** `min_load 0` (no GPU running when idle), `cold_workers 0` (nothing rented when idle),
   `max_workers 1`, `inactivity_timeout 300`. The first scan after an idle period waits for a 5–10 minute cold start.
   `cold_workers 1` would keep one stopped worker with the image and checkpoints on its disk (resumes in about a
   minute) at the cost of disk storage while it waits.
4. **Worker group:** one GPU with ≥ 12 GB and bf16 support (compute capability 8.x/9.x), a driver that supports CUDA ≥ 12.6, and a price cap of `dph_total<=0.20`.
   In practice that means an RTX 3060, A4000, 3090 or 4070 Super at about $0.06–0.17/hr. Without the cap, Vast picked an RTX 4090 at $0.45/hr.
   (A cap of 0.15 was too tight: the cheap cards were often taken and a $0.16/hr RTX 3090 was the best match.)
   The group also requires `static_ip=true` and excludes machines that failed (`machine_id nin [137275,150851]`), see below.

**Measured on Vast (every frame analysed):**

| Video | GPU | Decoder | Worker time |
|---|---|---|---|
| `000_003.mp4`, 16 s, 640×480, 396 frames | RTX 4090 | CPU (FF++ files are H.264 4:4:4, which NVDEC can't decode) | 2.4 s |
| 2 min 1440×1080 H.264 4:4:4, 3,168 frames | RTX A4000 ($0.09/hr) | CPU fallback | 54 s |
| 2 min 1440×1080 H.264 4:2:0 (typical upload), 3,168 frames | RTX A4000 ($0.09/hr) | NVDEC | **22 s** (decode fully hidden; detection 9 s; model 9 s) |

The first request after the endpoint scales to zero also waits for a cold start of about 5–10 minutes.

**Operating notes:**
- **Pin image tags.** Point the template at `cu126-<commit>`, not `cu126` or `latest`: Vast hosts cache tags that are re-pointed and can start an old image.
- **Replace workers after changing the template or image.** Updating the worker group's template recycles its workers. A recycled worker can keep a stale routing signature and answer every request with HTTP 401, so destroy the endpoint's workers (`vastai destroy instance <id> -y`) and let fresh ones start.
- **A worker that is "idle" but never receives scans is a bad host.** The worker log shows `num_requests_recieved: 0`
  while scans wait, and the client (with `debug=True`) logs `Worker unavailable (...)`. Seen so far:
  machine 137275 served a TLS certificate for a different IP than the router's URL (`ClientConnectorCertificateError`,
  "IP address mismatch"); machine 150851 had unreachable ports. Fix: add the machine to the `machine_id nin [...]`
  list in the worker group's search params, then `vastai destroy instance <id> -y` so a new worker starts elsewhere.
- **The Backblaze key lives in your Vast account environment variables** (`B2_KEY_ID`, `B2_APP_KEY`; set them with `scripts/set_vast_b2_env.sh`), not in the template.

Changing scale later takes one command (or the Vast console), for example `vastai update endpoint <id> --max_workers 2`.
When the Backblaze key is replaced, update the template's `B2_KEY_ID` / `B2_APP_KEY`; new workers pick it up.

## Local test (low-memory PCs)

```bash
cd model_server
ORIGINAI_WEIGHTS_DIR=../../models ORIGINAI_MODELS=veni-hq ORIGINAI_DEVICE=cpu OMP_NUM_THREADS=4 uvicorn app:app --port 18000
```

`ORIGINAI_WEIGHTS_DIR` points the server at the local checkpoints in the project's `models/` folder
(without it, it looks in `model_server/weights/`, where the Vast workers download them).

`ORIGINAI_MODELS` loads only the listed models. One Video Swin model on CPU uses about 1.5 GB of RAM and analyzes a 16-second clip in about a minute.

For Vidi (`ORIGINAI_MODELS=vidi`) also point `ORIGINAI_FAN_WEIGHTS` at the 2D-FAN file
(`2DFAN4-cd938726ad.zip`, see the Dockerfile) and use the pinned `av` version from `model_server/requirements.txt`.
The ViT-L/14 takes about 15 seconds per clip on a CPU, so a 16-second clip is 10 to 15 minutes.
