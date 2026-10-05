#!/bin/bash
# Vast.ai template on-start script:
#   1. download the checkpoints in models.json from Backblaze (verified by SHA-1),
#   2. start the model server,
#   3. start the Vast PyWorker (it watches $MODEL_LOG for readiness/errors).
export MODEL_LOG="${MODEL_LOG:-/var/log/originai/model.log}"
export TORCH_HOME="${TORCH_HOME:-/app/torch}"
mkdir -p "$(dirname "$MODEL_LOG")"

cd /app
(
  /opt/venv/bin/python fetch_weights.py &&
  exec /opt/venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 18000
) >> "$MODEL_LOG" 2>&1 &

curl -fsSL https://raw.githubusercontent.com/vast-ai/pyworker/main/start_server.sh | bash
