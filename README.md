# OriginAI on Vast.ai Serverless

Serves the four Video Swin-B checkpoints in `~/Desktop/Video Swin Models/` from one Vast.ai
Serverless endpoint. Vast does not turn a model into an API by itself: `model_server/` *is*
the API. Vast runs it on GPUs on demand and routes signed requests to it. The website's
`backend/server.py` is the only thing that talks to Vast, because every Vast call needs the
Vast API key.

```
Browser ──multipart──► website server.py ──JSON (base64 video)──► Vast route ──► GPU worker
        ◄── polls /api/analyses/{id}; loads evidence images from /api/analyses/{id}/media/…
                                                              PyWorker :3000 ─► model server :18000
```

| Model id  | Name    | Checkpoint             | Temperature T | Threshold* | Frame spacing Δ |
|-----------|---------|------------------------|---------------|------------|-----------------|
| `veni-hq` | Veni HQ | FF++ C23.pt            | 1.992         | 0.525      | 0.340 s |
| `veni-lq` | Veni LQ | FF++ C40.pt            | 2.559         | 0.489      | 0.340 s |
| `vidi`    | Vidi    | Celeb-DF v2.pt         | 5.016         | 0.229      | 0.245 s |
| `vici`    | Vici    | DeeperForensics-1.0.pt | 1.396         | 0.832      | 0.345 s |

\* The threshold applies to the calibrated probability `sigmoid(mean_logit / T)`, exactly as
in `Thesis_Code/src/swinfusionpp/p11_metric_stage.py`. Δ is the median per-video frame spacing
of each model's training split (`manifests/preprocessing_materialization_v001.parquet`).

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
- `status: "completed"`, `pred`, `prob_fake`, `raw_prob_fake`, `threshold`, `temperature`, `video_logit`
- `clips[{index, start_s, end_s, logit, prob, clips, heatmap}]` (timeline windows)
- `protocol{available, pred, prob_fake, clip_logits | reason, message}`
- `coverage{mode, decoder, device, frames_decoded, frames_analyzed, face_frames, phases, frame_spacing_s, clips, analyzed_start_s, analyzed_end_s, fallback, …}`
- `heatmaps{target, frame_size, stops, clips[{clip_index, role, clip_start_s, clip_end_s, clip_prob, frames[{t, frame_jpg, frame_heat_jpg, crop_jpg, crop_heat_jpg}]}]}` (base64 JPEGs, about 2 MB)
- `timings`

**Failure** returns `{"status": "failed", "reason", "error"}`, where `reason` is one of:
`no_face`, `low_detection_confidence`, `identity_track_failed`, `insufficient_valid_frames`, `decode_failed`, `metadata_error`.

**Proxy.** The website proxy (`backend/server.py`) exposes `POST /api/analyses`, `GET /api/analyses/{id}` and `GET /api/analyses/{id}/media/{file}`. It writes the heatmap images to `backend/outputs/analyses/<id>/`, replaces the base64 with URLs, and deletes them after an hour.

## Deploy

Repository layout (this repo is public and holds code only):

- `worker.py`, `requirements.txt`: the Vast PyWorker. The template's `PYWORKER_REPO` points at this repo.
- `model_server/`: the model API (`app.py`), detector (`detector.py`), registry (`models.json`) and weight downloader (`fetch_weights.py`).
- `Dockerfile`, `onstart.sh`: the worker image, built by `.github/workflows/build-image.yml` into `ghcr.io/<owner>/originai-worker`.

The four checkpoints stay private in the Backblaze bucket. Each worker downloads them at start with a read-only key (`B2_KEY_ID`, `B2_APP_KEY` in the Vast template) and checks every file's SHA-1 against `models.json`.

1. **Image:** pushing to `main` builds `:latest` / `:cu126`. For RTX 50-series hosts, run the workflow manually with `cu130`.
2. **Vast template:**
   - image: `ghcr.io/<owner>/originai-worker:cu126`
   - Docker options: `-p 3000:3000 -e PYWORKER_REPO=<this repo> -e B2_KEY_ID=… -e B2_APP_KEY=…`
   - on-start: `bash /onstart.sh`
3. **Endpoint, lowest-cost test settings:** `min_load 0` (scale to zero), `cold_workers 0`, `max_workers 1`, `inactivity_timeout 300`.
4. **Worker group:** one GPU with ≥ 12 GB and bf16 support (compute capability 8.x/9.x), a driver that supports CUDA ≥ 12.6, and a price cap of `dph<=0.15`.
   In practice that means an RTX 3060 or A4000 at about $0.06–0.10/hr. Without the cap, Vast picked an RTX 4090 at $0.45/hr.

**Measured on Vast (RTX 4090, `000_003.mp4`):**
- **Result:** DEEPFAKE 99.76%, 3-clip logits 11.88 / 11.86 / 12.26 (training recorded 11.63 / 11.94 / 12.31).
- **Speed:** the analysis itself takes about 2 s.
- **First cold start:** about 13 minutes, covering the GPU rental, the image pull and the model download from Backblaze. The first run also spent about 5 minutes installing OpenSSH, which the image now preinstalls.
5. **Website server:** `VAST_API_KEY=<key> VAST_ENDPOINT=originai-detectors uvicorn server:app --port 8000`.
   If the static site is hosted elsewhere, set `window.ORIGINAI_API_BASE = "https://<api-host>"` before `scanner.js` loads.

Changing scale later takes one command (or the Vast console), for example `vastai update endpoint <id> --max_workers 2`.
When the Backblaze key is replaced, update the template's `B2_KEY_ID` / `B2_APP_KEY`; new workers pick it up.

## Local test (low-memory PCs)

```bash
cd model_server
ORIGINAI_MODELS=veni-hq ORIGINAI_DEVICE=cpu OMP_NUM_THREADS=4 uvicorn app:app --port 18000
```

`ORIGINAI_MODELS` loads only the listed models. One model on CPU uses about 1.5 GB of RAM and analyzes a 16-second clip in about a minute.
