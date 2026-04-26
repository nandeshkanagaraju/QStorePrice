# QStorePrice OpenEnv server image.
#
# Builds the Vite React sim UI (`web/`) and serves it at /sim/ on the same
# origin as POST /api/sim/* (FastAPI), so the Space behaves like local
# `vite` + proxy. Used by the Hugging Face Space (sdk: docker):
#
#     docker run -d -p 8000:8000 registry.hf.space/<user>-qstoreprice:latest
#
# Root `/` redirects to `/sim/` when SIM_UI_DEFAULT=1 (default here). Set
# SIM_UI_DEFAULT=0 to serve the legacy HTML KPI dashboard at `/` instead.
#
# Gradio demo: override CMD, e.g. `docker run ... python app.py`
# ---------------------------------------------------------------------------
FROM node:22-bookworm-slim AS sim-ui
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
ENV VITE_BASE=/sim/
RUN npm run build

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

# React sim dashboard (same API paths as local: /api/sim/reset, /api/sim/step).
COPY --from=sim-ui /src/web/dist ./static/sim

# Install the env as a package so `from models import ...` and
# `from client import ...` work the way the OpenEnv CLI expects.
RUN pip install --no-cache-dir -e .

# Default port matches README Space frontmatter (app_port: 8000).
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=info \
    SIM_UI_DEFAULT=1 \
    ENABLE_WEB_INTERFACE=false

EXPOSE 8000

# Healthcheck so HF Space + orchestrators can detect a wedged container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Default: launch the OpenEnv FastAPI server.
CMD ["python", "-m", "server.app"]
