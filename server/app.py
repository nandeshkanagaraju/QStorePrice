"""FastAPI server for the QStorePrice OpenEnv environment.

OpenEnv is used as the deployment / contract layer for the Hugging Face
Space that hosts the live demo of our Gemma 4 Good Hackathon submission.
Exposes the canonical OpenEnv HTTP/WebSocket endpoints:

    GET  /health   - liveness probe
    POST /reset    - start a new episode
    POST /step     - submit an Operating Brief, advance the simulation
    GET  /state    - current FreshPriceState snapshot
    WS   /ws       - persistent session used by the async client
    GET  /docs     - OpenAPI documentation

Plus admin / dashboard endpoints (additive — do not affect the OpenEnv
contract):

    GET  /admin/dashboard            - live metrics snapshot (JSON)
    GET  /admin/metrics/scores       - flat list of episode records
    GET  /admin/metrics/reward-curve - flat list of step records
    GET  /admin/tasks                - curriculum scenario list
    POST /admin/metrics/reset        - clear in-memory metrics
    GET  /                           - redirects to /sim/ when Docker-built UI exists
                                       and SIM_UI_DEFAULT=1; else legacy KPI HTML
    GET  /sim/                       - FreshQuick React sim (built in Docker image)
    GET  /kpi                        - legacy HTML KPI dashboard
    GET  /dashboard                  - same dashboard when HF web UI is enabled

The OpenEnv app uses openenv-core's `create_app` (standard API; Gradio at
``/web`` when ``ENABLE_WEB_INTERFACE`` is set, e.g. after ``openenv push``).
If openenv-core is missing, the server falls back to a plain FastAPI app.

Run:

    # Local development (with reload)
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

    # Production (matches the Dockerfile CMD)
    python -m server.app
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from openenv.core.env_server import create_app
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False
    create_app = None  # type: ignore[assignment]


def _web_interface_enabled() -> bool:
    return os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _prefer_sim_ui_at_root() -> bool:
    """When true and ``static/sim/index.html`` exists, ``GET /`` redirects to ``/sim/``."""
    return os.environ.get("SIM_UI_DEFAULT", "").lower() in ("true", "1", "yes")


def _build_app():
    """Build the FastAPI app — falls back to a plain FastAPI() if openenv-core
    is missing so the dashboard still works in dev environments."""
    if _OPENENV_AVAILABLE:
        from server.environment import BriefAction, BriefObservation, FreshPriceOpenEnv
        return create_app(
            env=FreshPriceOpenEnv,
            action_cls=BriefAction,
            observation_cls=BriefObservation,
        )

    from fastapi import FastAPI
    fallback = FastAPI(title="QStorePrice (fallback - openenv-core missing)")

    @fallback.get("/health")
    def _health():
        return {"status": "ok", "openenv_core": False}

    return fallback


app = _build_app()

# CORS for local Vite dev server (React sim dashboard)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

_origins = os.environ.get(
    "SIM_UI_ORIGINS",
    "http://127.0.0.1:5173,http://localhost:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from server.demo_sim import router as demo_sim_router  # noqa: E402

app.include_router(demo_sim_router)


# ---------------------------------------------------------------------------
# Admin / dashboard endpoints (additive — do not change OpenEnv contract)
# ---------------------------------------------------------------------------

from freshprice_env.enums import CurriculumScenario  # noqa: E402
from freshprice_env.monitoring import metrics  # noqa: E402


@app.get("/admin/dashboard", tags=["Admin"])
def admin_dashboard():
    """Full metrics snapshot — feeds the live HTML dashboard."""
    return metrics.get_dashboard()


@app.get("/admin/metrics/scores", tags=["Admin"])
def admin_metrics_scores(scenario: str | None = None):
    return {"episodes": metrics.get_episode_scores(scenario=scenario)}


@app.get("/admin/metrics/reward-curve", tags=["Admin"])
def admin_metrics_reward_curve(scenario: str | None = None):
    return {"steps": metrics.get_reward_curve(scenario=scenario)}


@app.get("/admin/tasks", tags=["Admin"])
def admin_tasks():
    """Curriculum scenarios available to the environment."""
    return {
        "tasks": [
            {"level": s.value, "name": s.name}
            for s in CurriculumScenario
        ]
    }


@app.post("/admin/metrics/reset", tags=["Admin"])
def admin_metrics_reset():
    metrics.reset()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Static dashboard (optional)
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles  # noqa: E402
    from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    _sim_ui_dir = _STATIC_DIR / "sim"
    if (_sim_ui_dir / "index.html").is_file():
        app.mount(
            "/sim",
            StaticFiles(directory=str(_sim_ui_dir), html=True),
            name="sim_ui",
        )

    if not _web_interface_enabled():

        @app.get("/kpi", include_in_schema=False)
        def kpi_dashboard():
            """Original polling KPI dashboard (HTML + /static/*)."""
            return FileResponse(str(_STATIC_DIR / "index.html"))

        @app.get("/", include_in_schema=False)
        def dashboard_index():
            if (_sim_ui_dir / "index.html").is_file() and _prefer_sim_ui_at_root():
                return RedirectResponse(url="/sim/", status_code=302)
            return FileResponse(str(_STATIC_DIR / "index.html"))
    else:

        @app.get("/dashboard", include_in_schema=False)
        def dashboard_index_hf():
            """HTML KPI dashboard (root redirects to OpenEnv /web/)."""
            return FileResponse(str(_STATIC_DIR / "index.html"))


def main() -> None:
    """CLI entry point used by `python -m server.app` and the Dockerfile."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
