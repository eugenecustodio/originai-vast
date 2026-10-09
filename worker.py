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
        on_error=["Traceback (most recent call last):", "Application startup failed"],
        on_info=["Loaded "],
    ),
)

Worker(worker_config).run()
