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

## Preprocessing: the frozen thesis contract

Ported from `Thesis_Code/src/swinfusionpp/preprocessing.py` and
`configs/preprocessing/policy_v001.yaml` (policy `preprocessing_v001`), and stored in
`model_server/models.json → policy`. Do not change these values.

- **Decode:** PyAV 16.0.1, taking the first decoded frame at or after each requested timestamp.
- **Faces:** facenet-pytorch 2.6.0 MTCNN (`min_face_size=40`, thresholds `[0.6, 0.7, 0.7]`, `factor=0.709`), keeping detections with probability ≥ 0.90.
- **Alignment:** five-landmark similarity transform to the ArcFace template, 224×224 RGB uint8, linear interpolation, reflect-101 padding.
- **Identity tracking:** FaceNet vggface2 embeddings (cosine distance ≤ 0.55 or IoU ≥ 0.10, ambiguity margin 0.05, EMA 0.9).
- **Model:** torchvision `swin3d_b` fed 16 frames (B×16×3×224×224). ImageNet mean/std is applied inside the checkpoint's normalizer, then `Linear(1024 → 1)`.

**Verified against training.** Your test video `000_003.mp4` is in the FF++ C23 test split:
- **Frame selection:** our 48 selected frames have the same timestamps and decoded-frame indices as the training cache.
- **Face crops:** re-aligned with training's recorded landmarks, the crops are bit-identical (48/48).
- **Landmarks:** MTCNN on CPU differs from the GPU landmarks by 0.07 px on average.
- **Logits:** the 3-clip logits are 11.79 / 11.94 / 12.27, against 11.63 / 11.94 / 12.31 recorded in training (bf16).

## Whole-video analysis (the model only sees 16 frames)

1. **Whole video (drives the verdict).**
   - Frames are sampled every Δ seconds from 2% to 98% of the duration, which keeps each clip at the frame spacing the model was trained on, even for a 2-minute video.
   - The primary face is tracked and the track is tiled into 16-frame clips with a stride of 8 (50% overlap), plus a final clip flush with the end.
   - Clips never bridge a gap longer than 4Δ, for example when the face leaves the shot.
   - Verdict: mean of all clip logits → `sigmoid(mean / T)` → threshold. That is the thesis read-out.
2. **Benchmark protocol (shown next to it).**
   - The exact thesis evaluation: 64 timestamps → track → 48 frames → three anchors `[0:16] [16:32] [32:48]`.
   - It is reported as "Benchmark protocol · 3 clips", or "Not available" if fewer than 48 face frames were tracked.
   - If whole-video tiling finds too few face frames but this protocol works, its anchors are used for the verdict instead (`coverage.fallback`).

Portrait phone videos are rotated upright before face detection.

## Evidence maps ("where the model looked")

**How they're computed:**
- The Video Swin-B head is LayerNorm → average pool → linear.
- So each final-stage token (8×7×7 per clip) contributes exactly `w · LayerNorm(token)` to the clip logit, and the mean of those contributions plus the bias equals the logit (checked to 1e-6).
- This class activation map equals Grad-CAM taken at the final norm layer. It needs no backward pass and costs about 2 s extra on CPU.

**What's rendered:**
- Maps are oriented toward the verdict and made for the 4 most suspicious clips plus the least suspicious one.
- They are scaled together, relative within the video: at or below the median is clear, the 99.5th percentile is full strength.
- Each clip shows 8 frames, the first frame of each 2-frame tubelet, over both the original video frame and the 224×224 face crop the model saw.

**Limits:**
- The resolution is 7×7 per tubelet, so maps show regions, not pixels.
- Brightness is relative, so it's a guide, not a measurement.
- Clip probabilities are calibrated with the video-level temperature, so they're approximate per clip.

## Worker API (`model_server/app.py`)

`POST /analyze` takes `{"model": id, "video_b64" | "video_url", "filename"}`.

**Success** returns:
- `status: "completed"`, `pred`, `prob_fake`, `raw_prob_fake`, `threshold`, `temperature`, `video_logit`
- `clips[{index, start_s, end_s, logit, prob, heatmap}]`
- `protocol{available, pred, prob_fake, clip_logits | reason, message}`
- `coverage{duration, fps, width, height, frame_spacing_s, face_frames, segments, clips, clip_stride, analyzed_start_s, analyzed_end_s, fallback}`
- `heatmaps{target, frame_size, stops, clips[{clip_index, role, frames[{t, frame_jpg, frame_heat_png, crop_jpg, crop_heat_png}]}]}` (base64 images, about 5 MB)
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
4. **Worker group:** the cheapest reliable single GPU with ≥ 12 GB and a driver that supports CUDA ≥ 12.6.
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
