# QStorePrice OpenEnv server image.
#
# Builds a small image that serves the FastAPI OpenEnv app on port 8000 (the
# OpenEnv default — PDF page 47 / page 48). Used by the Hugging Face Space
# (sdk: docker) and by anyone running `docker run` locally:
#
#     docker run -d -p 8000:8000 registry.hf.space/<user>-qstoreprice:latest
#
# To run the optional Gradio demo UI inside the same image, override the CMD:
#     docker run -e PORT=7860 ... python app.py
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal so the image stays small (PDF page 13 note about
# avoiding big files in env submissions).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps. We install the lightweight server requirements first
# so the layer is cacheable across rebuilds; training-only packages (torch,
# trl, unsloth) live in requirements_training.txt and are NOT installed here.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "openenv-core>=0.2.0" "fastapi>=0.110" "uvicorn>=0.27"

# Copy the package itself
COPY . .

# Install the env as a package so `from models import ...` and
# `from client import ...` work the way the OpenEnv CLI expects.
RUN pip install --no-cache-dir -e .

# Default port matches the OpenEnv canonical server (PDF page 47).
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=info

EXPOSE 8000

# Healthcheck so HF Space + orchestrators can detect a wedged container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Default: launch the OpenEnv FastAPI server.
CMD ["python", "-m", "server.app"]
