"""OriginAI model server - runs on each Vast.ai serverless worker behind the PyWorker.

Routes (JSON only; the Vast PyWorker forwards JSON payloads):
  GET  /health
  GET  /models
  POST /analyze   {"model": "<id>", "video_b64": "<base64>" | "video_url": "https://...",
                   "filename": "clip.mp4"}
                  {"benchmark": true, "model": "<id>"}  -> synthetic GPU pass (used by the
                   PyWorker readiness benchmark; no video needed)

Analysis failures (no face, undecodable video) return HTTP 200 with
{"status": "failed", "reason": <code>, "error": <message>} so the Vast client does not retry them.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from detector import AnalysisFailed, DetectorService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("originai")

APP_DIR = Path(__file__).resolve().parent
REGISTRY = Path(os.environ.get("ORIGINAI_REGISTRY", APP_DIR / "models.json"))
WEIGHTS_DIR = Path(os.environ.get("ORIGINAI_WEIGHTS_DIR", APP_DIR / "weights"))
DEVICE = os.environ.get("ORIGINAI_DEVICE", "auto")
# Optional comma-separated subset of model ids to load (e.g. on a low-memory test machine).
ONLY_MODELS = [m.strip() for m in os.environ.get("ORIGINAI_MODELS", "").split(",") if m.strip()] or None
MAX_VIDEO_BYTES = int(os.environ.get("ORIGINAI_MAX_VIDEO_MB", "150")) * 1024 * 1024
ALLOWED_EXTS = {".mp4", ".mov", ".webm", ".mkv"}

app = FastAPI(title="OriginAI model server")
service: DetectorService | None = None


@app.on_event("startup")
def _load() -> None:
    global service
    t = time.time()
    service = DetectorService(REGISTRY, WEIGHTS_DIR, DEVICE, only=ONLY_MODELS)
    log.info("Loaded %d models on %s in %.1fs", len(service.models), service.device, time.time() - t)


class AnalyzeRequest(BaseModel):
    model: str
    video_b64: str | None = None
    video_url: str | None = None
    filename: str | None = None
    benchmark: bool = False


@app.get("/health")
def health() -> dict:
    return {
        "status": "ready" if service else "loading",
        "device": str(service.device) if service else None,
        "models": list(service.models) if service else [],
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

    t = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        video_path = _write_video(req, Path(tmp))
        try:
            result = service.analyze(video_path, req.model)
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
        "Analyzed with %s: %s p=%.4f frames=%s clips=%s decoder=%s in %.1fs %s",
        req.model, result["pred"], result["prob_fake"], coverage.get("frames_analyzed"),
        coverage.get("clips"), coverage.get("decoder"), result["seconds"], result.get("timings"),
    )
    return result
