"""OriginAI model server - runs on each Vast.ai serverless worker behind the PyWorker.

Routes (JSON only; the Vast PyWorker forwards JSON payloads):
  GET  /health
  GET  /models
  POST /analyze   {"model": "<id>", "video_b64": "<base64>" | "video_url": "https://...",
                   "filename": "clip.mp4", "start_s": 12.0, "end_s": 20.0}
                  start_s / end_s are optional: the part of the video to check (whole video if left out)
                  {"benchmark": true, "model": "<id>"}  -> synthetic GPU pass (used by the
                   PyWorker readiness benchmark; no video needed)

Analysis failures (no face, undecodable video) return HTTP 200 with
{"status": "failed", "reason": <code>, "error": <message>} so the Vast client does not retry them.
"""

from __future__ import annotations

import base64
import binascii
import logging
import math
import os
import tempfile
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Several scans share the GPU: let PyTorch's caching allocator grow and shrink its memory
# segments instead of keeping fixed blocks, which avoids most fragmentation (set before torch loads).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# cuBLAS may pick different internal implementations when several streams are in use, which
# makes results vary from run to run; a fixed workspace keeps every scan's numbers reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from detector import AnalysisFailed, DetectorService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("originai")

APP_DIR = Path(__file__).resolve().parent
REGISTRY = Path(os.environ.get("ORIGINAI_REGISTRY", APP_DIR / "models.json"))
WEIGHTS_DIR = Path(os.environ.get("ORIGINAI_WEIGHTS_DIR", APP_DIR / "weights"))
DEVICE = os.environ.get("ORIGINAI_DEVICE", "auto")
# Optional comma-separated subset of model ids to load (e.g. on a low-memory test machine).
ONLY_MODELS = [m.strip() for m in os.environ.get("ORIGINAI_MODELS", "").split(",") if m.strip()] or None
MAX_VIDEO_BYTES = int(os.environ.get("ORIGINAI_MAX_VIDEO_MB", "250")) * 1024 * 1024
ALLOWED_EXTS = {".mp4", ".mov", ".webm", ".mkv"}

app = FastAPI(title="OriginAI model server")
service: DetectorService | None = None


@app.on_event("startup")
def _load() -> None:
    global service
    t = time.time()
    service = DetectorService(REGISTRY, WEIGHTS_DIR, DEVICE, only=ONLY_MODELS)
    log.info("Loaded %d models on %s in %.1fs; up to %d scans at once (%d in the model pass)",
             len(service.models), service.device, time.time() - t, service.gate.slots, service.heavy_slots)


class AnalyzeRequest(BaseModel):
    model: str
    video_b64: str | None = None
    video_url: str | None = None
    filename: str | None = None
    benchmark: bool = False
    start_s: float | None = None  # the part of the video to check; both or neither
    end_s: float | None = None


def _window(req: AnalyzeRequest) -> tuple[float, float] | None:
    if req.start_s is None and req.end_s is None:
        return None
    if req.start_s is None or req.end_s is None:
        raise HTTPException(400, "Give both start_s and end_s, or neither.")
    if not (math.isfinite(req.start_s) and math.isfinite(req.end_s)) or req.start_s < 0 or req.end_s <= req.start_s:
        raise HTTPException(400, "start_s and end_s must be seconds with 0 <= start_s < end_s.")
    return float(req.start_s), float(req.end_s)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ready" if service else "loading",
        "device": str(service.device) if service else None,
        "models": list(service.models) if service else [],
        "scans": service.status() if service else None,
    }


@app.get("/models")
def models() -> dict:
    if service is None:
        raise HTTPException(503, "Models are still loading.")
    return {"models": [m.public_info() for m in service.models.values()]}


def _write_video(req: AnalyzeRequest, dest_dir: Path) -> Path:
    suffix = Path(req.filename or req.video_url or "upload.mp4").suffix.lower().split("?")[0]
    if suffix not in ALLOWED_EXTS:
        suffix = ".mp4"
    path = dest_dir / f"input{suffix}"
    if req.video_b64:
        try:
            data = base64.b64decode(req.video_b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(400, "video_b64 is not valid base64.")
        if len(data) > MAX_VIDEO_BYTES:
            raise HTTPException(413, "Video is too large.")
        path.write_bytes(data)
    elif req.video_url:
        if not req.video_url.startswith(("https://", "http://")):
            raise HTTPException(400, "video_url must be http(s).")
        try:
            with urllib.request.urlopen(req.video_url, timeout=60) as resp, path.open("wb") as f:
                written = 0
                while chunk := resp.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_VIDEO_BYTES:
                        raise HTTPException(413, "Video is too large.")
                    f.write(chunk)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"Could not download video_url: {exc}")
    else:
        raise HTTPException(400, "Provide video_b64 or video_url.")
    return path


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    if service is None:
        raise HTTPException(503, "Models are still loading.")
    if req.model not in service.models:
        raise HTTPException(400, f"Unknown model '{req.model}'. Available: {', '.join(service.models)}")

    if req.benchmark:
        try:
            result = service.benchmark(req.model)
        except Exception as exc:  # logged without a traceback so the PyWorker does not mark the worker dead
            log.error("Benchmark error: %r", exc)
            raise HTTPException(500, "Benchmark failed on the model server.")
        return {"status": "completed", "benchmark": True, **result}

    window = _window(req)
    t = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        video_path = _write_video(req, Path(tmp))
        try:
            result = service.analyze(video_path, req.model, window)
        except AnalysisFailed as exc:
            log.info("Analysis failed: %s (%s)", exc.reason, exc)
            return {"status": "failed", "reason": exc.reason, "error": str(exc)}
        except Exception as exc:
            log.error("Inference error: %r", exc)
            raise HTTPException(500, "Inference failed on the model server.")
    result["status"] = "completed"
    result["seconds"] = round(time.time() - t, 2)
    coverage = result.get("coverage", {})
    log.info(
        "Analyzed with %s%s: %s p=%.4f frames=%s clips=%s decoder=%s in %.1fs %s",
        req.model, f" ({window[0]:.1f}-{window[1]:.1f} s)" if window else "", result["pred"], result["prob_fake"], coverage.get("frames_analyzed"),
        coverage.get("clips"), coverage.get("decoder"), result["seconds"], result.get("timings"),
    )
    return result
