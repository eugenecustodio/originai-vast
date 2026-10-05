# syntax=docker/dockerfile:1
# OriginAI deepfake detectors (Video Swin-B and DFD-FCG) for Vast.ai Serverless.
# Built by .github/workflows/build-image.yml and pushed to ghcr.io/<owner>/originai-worker.
# The image holds code only; model checkpoints are downloaded from the private Backblaze
# bucket at worker start (model_server/fetch_weights.py).
# RTX 50-series (Blackwell) hosts need the CUDA 13.0 build (host driver >= 580): TORCH_CUDA=cu130.
FROM ubuntu:24.04

# NVIDIA_DRIVER_CAPABILITIES must include "video" so the container runtime mounts the
# driver's NVDEC library (libnvcuvid) used for GPU video decoding.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    TORCH_HOME=/app/torch \
    MODEL_LOG=/var/log/originai/model.log \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# python3.12 is Ubuntu 24.04's system Python (the thesis used 3.12).
# git/curl/openssl are required by Vast's PyWorker start script. openssh-server is
# preinstalled so Vast's SSH launch mode does not spend minutes installing it on
# every new host (it added ~5 min to the first cold start). ffmpeg provides the
# shared libraries torchcodec uses for demuxing and CPU-fallback decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv git curl openssl ca-certificates libglib2.0-0 ffmpeg \
    && apt-get install -y openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ARG TORCH_CUDA=cu126
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
        torch==2.12.1 torchvision==0.27.1 torchcodec==0.16.0

COPY model_server/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && pip install --no-deps facenet-pytorch==2.6.0

# FaceNet identity weights used by the thesis tracker, verified against the training registry.
RUN mkdir -p /app/torch/checkpoints \
    && curl -fsSL -o /app/torch/checkpoints/20180402-114759-vggface2.pt \
       https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt \
    && echo "281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b  /app/torch/checkpoints/20180402-114759-vggface2.pt" | sha256sum -c -

# 2D-FAN landmark model (face_alignment 1.4.1, TorchScript): DFD-FCG's face crop is built from its landmarks.
RUN curl -fsSL --retry 3 -o /app/torch/checkpoints/2DFAN4-cd938726ad.zip \
       https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip \
    && echo "cd938726adb1f15f361263cce2db9cb820c42585fa8796ec72ce19107f369a46  /app/torch/checkpoints/2DFAN4-cd938726ad.zip" | sha256sum -c -

COPY model_server/ /app/
COPY onstart.sh /onstart.sh
RUN chmod +x /onstart.sh && mkdir -p /var/log/originai /app/weights && cd /app && python - <<'EOF'
import hashlib, json, os
import facenet_pytorch
from torchvision.models.video import swin3d_b  # noqa: F401
import detector, dfd_fcg, gpu_mtcnn, video_io  # noqa: F401  (import cleanly)
from torchcodec.decoders import VideoDecoder, set_cuda_backend  # noqa: F401
# MTCNN weights must match the thesis preprocessing registry.
expected = {
    "pnet.pt": "a2a71925e0b9996a42f63e47efc1ca19043e69558b5c523b978d611dfae49c8f",
    "rnet.pt": "bbb937de72efc9ef83b186c49f5f558467a1d7e3453a8ece0d71a886633f6a86",
    "onet.pt": "165bfbe42940416ccfb977545cf0e976d5bf321f67083ae2aaaa5c764280118d",
}
data = os.path.join(os.path.dirname(facenet_pytorch.__file__), "data")
for name, digest in expected.items():
    assert hashlib.sha256(open(os.path.join(data, name), "rb").read()).hexdigest() == digest, name
registry = json.load(open("models.json"))
for model in registry["models"]:
    assert model["b2_path"] and len(model["sha1"]) == 40, model["id"]
# The landmark model in the image is the one models.json names.
landmarks = registry["dfd_fcg"]["landmarks"]
path = dfd_fcg.landmark_weights("/app/weights", landmarks["file"])
assert hashlib.sha256(open(path, "rb").read()).hexdigest() == landmarks["sha256"], path
print("image checks passed")
EOF
