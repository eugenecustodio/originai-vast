"""Download the checkpoints listed in models.json from the private Backblaze B2 bucket.

Runs on each Vast worker before the model server starts (see onstart.sh). Files that
already exist with the right SHA-1 are skipped; every download is verified against the
SHA-1 recorded in models.json, so the worker serves exactly the local checkpoint files.

Env: B2_KEY_ID, B2_APP_KEY (a read-only key for the bucket), optional ORIGINAI_MODELS.
On failure it prints "Application startup failed: ..." so the Vast PyWorker marks the
worker as errored instead of waiting forever.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REGISTRY = Path(os.environ.get("ORIGINAI_REGISTRY", APP_DIR / "models.json"))
WEIGHTS_DIR = Path(os.environ.get("ORIGINAI_WEIGHTS_DIR", APP_DIR / "weights"))
ONLY = [m.strip() for m in os.environ.get("ORIGINAI_MODELS", "").split(",") if m.strip()]


def sha1_of(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def authorize() -> dict:
    key_id, app_key = os.environ.get("B2_KEY_ID", ""), os.environ.get("B2_APP_KEY", "")
    if not key_id or not app_key:
        raise RuntimeError("B2_KEY_ID / B2_APP_KEY are not set")
    token = base64.b64encode(f"{key_id}:{app_key}".encode()).decode()
    request = urllib.request.Request(
        "https://api.backblazeb2.com/b2api/v3/b2_authorize_account",
        headers={"Authorization": "Basic " + token},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download(auth: dict, bucket: str, model: dict) -> str:
    target = WEIGHTS_DIR / model["checkpoint"]
    url = (
        auth["apiInfo"]["storageApi"]["downloadUrl"]
        + f"/file/{bucket}/"
        + urllib.parse.quote(model["b2_path"])
    )
    for attempt in range(1, 4):
        partial = target.with_name(target.name + ".part")
        try:
            started = time.time()
            request = urllib.request.Request(url, headers={"Authorization": auth["authorizationToken"]})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as f:
                while chunk := response.read(1 << 22):
                    f.write(chunk)
            actual = sha1_of(partial)
            if actual != model["sha1"]:
                raise RuntimeError(f"SHA-1 mismatch for {model['checkpoint']}: {actual}")
            partial.replace(target)
            size = target.stat().st_size / 1e6
            return f"{model['checkpoint']}: {size:.0f} MB in {time.time() - started:.1f}s (sha1 ok)"
        except Exception as exc:  # retry network hiccups
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"{model['checkpoint']}: {exc}") from None
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def main() -> int:
    cfg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    models = [m for m in cfg["models"] if not ONLY or m["id"] in ONLY]
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for model in models:
        path = WEIGHTS_DIR / model["checkpoint"]
        if path.exists() and sha1_of(path) == model["sha1"]:
            print(f"weights: {model['checkpoint']} already present (sha1 ok)", flush=True)
        else:
            missing.append(model)
    if not missing:
        return 0
    try:
        auth = authorize()
        bucket = cfg["weights_source"]["bucket"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for line in pool.map(lambda m: download(auth, bucket, m), missing):
                print(f"weights: {line}", flush=True)
    except Exception as exc:
        # Printed without a traceback; this exact prefix is a PyWorker on_error trigger.
        print(f"Application startup failed: could not download model weights ({exc})", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
