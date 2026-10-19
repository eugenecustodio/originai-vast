"""Vast.ai PyWorker for the OriginAI model server.

Push this folder (worker.py + requirements.txt) to a PUBLIC Git repo and set the
template env var PYWORKER_REPO to its URL. The PyWorker proxies signed requests
to the model server started by onstart.sh on port 18000.
"""

import os

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=18000,
    model_log_file=os.environ.get("MODEL_LOG", "/var/log/originai/model.log"),
    model_healthcheck_url="/health",
    handlers=[
        HandlerConfig(
            route="/analyze",
            # Requests go straight to the model server, which runs several videos at once on the
            # GPU and queues the rest first come, first served (ScanGate in model_server/detector.py).
            allow_parallel_requests=True,
            max_queue_time=600.0,
            # Longer videos take longer (whole-video tiling); a constant per-request cost keeps
            # the autoscaler's view simple.
            workload_calculator=lambda payload: 100.0,
            benchmark_config=BenchmarkConfig(
                # Synthetic GPU pass - no video or face detection needed.
                dataset=[{"model": m, "benchmark": True} for m in ("veni-hq", "veni-lq", "vidi", "vici")],
                runs=4,
                concurrency=int(os.environ.get("ORIGINAI_CONCURRENCY", "0") or 0) or 2,
            ),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["Application startup complete."],
        # Only a failed start is fatal (uvicorn and fetch_weights.py both print this). Any traceback used to count
        # too, so one request that hit an unexpected error (e.g. a video download cut off midway) marked the whole
        # worker as broken for the rest of its life and Vast replaced it: a 15-30 minute cold start for everyone.
        # A model server that really stops answering is caught by the /health check above.
        # Also fatal: a host whose GPU PyTorch can't use (the server would load on the CPU, report ready and time
        # out every scan), and CUDA errors that break the GPU context for good (every later scan fails or waits
        # forever while /health stays green). Not "CUDA error" alone: an out-of-memory retry can log
        # "CUDA error: CUBLAS_STATUS_ALLOC_FAILED", which the server recovers from.
        on_error=[
            "Application startup failed",
            "models on cpu",
            "illegal memory access",
            "device-side assert",
            "unspecified launch failure",
            "misaligned address",
            "uncorrectable ECC error",
            "CUDA-capable device",
            "CUDA error: unknown error",
        ],
        on_info=["Loaded "],
    ),
)

Worker(worker_config).run()
