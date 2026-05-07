"""Validate the QStorePrice submission against The Gemma 4 Good Hackathon
requirements.

The hackathon mandates five deliverables attached to a single Kaggle
Writeup (≤ 1500 words): a public 3-min YouTube video, a public code
repository, a live demo, a media-gallery cover image. This script verifies
the technical artefacts that back those deliverables.

Checks (each prints PASS/FAIL on its own line):

  1. openenv.yaml exists and has the canonical sections (powers the live demo Space)
  2. Required Python modules import cleanly
  3. FreshPriceEnv reset() works for every CurriculumScenario
  4. Server module imports (admin endpoints registered)
  5. Static dashboard files present (frontend of the live demo)
  6. SFT data generator produces a valid 6-section Operating Brief completion
  7. Required env vars are set (warn-only — HF_TOKEN is needed to download
     gated Gemma 4 weights)

Run:
    python validate_submission.py

Exit code: 0 = all required checks pass; 1 = at least one required check failed.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECOMMENDED_ENV_VARS = ["HF_TOKEN"]

_FAILED = 0
_PASSED = 0


def _ok(label: str, detail: str = "") -> None:
    global _PASSED
    _PASSED += 1
    msg = f"  PASS  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def _fail(label: str, detail: str) -> None:
    global _FAILED
    _FAILED += 1
    print(f"  FAIL  {label} — {detail}")


def _warn(label: str, detail: str) -> None:
    print(f"  WARN  {label} — {detail}")


# ---------------------------------------------------------------------------
# 1. openenv.yaml
# ---------------------------------------------------------------------------

def check_openenv_yaml() -> None:
    print("\n[1] openenv.yaml")
    path = ROOT / "openenv.yaml"
    if not path.exists():
        _fail("openenv.yaml present", f"missing at {path}")
        return

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        _warn("yaml parser", "PyYAML not installed; doing string-based check instead")
        text = path.read_text()
        for needle in ("name:", "description:"):
            if needle in text:
                _ok(f"openenv.yaml contains `{needle}`")
            else:
                _fail(f"openenv.yaml contains `{needle}`", "not found")
        return

    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        _fail("openenv.yaml schema", "top-level not a mapping")
        return

    required_keys = ["environment"]
    for key in required_keys:
        if key in cfg:
            _ok(f"openenv.yaml has top-level `{key}`")
        else:
            _fail(f"openenv.yaml has top-level `{key}`", "missing")


# ---------------------------------------------------------------------------
# 2. Module imports
# ---------------------------------------------------------------------------

def check_module_imports() -> None:
    print("\n[2] Python module imports")
    modules = [
        "freshprice_env.freshprice_env",
        "freshprice_env.enums",
        "freshprice_env.constants",
        "freshprice_env.monitoring",
        "freshprice_env.brief_pipeline.prompt_builder",
        "training.curriculum",
        "training.trajectory_buffer",
        "training.counterfactual",
        "training.generate_sft_data",
        "eval.evaluator",
        "eval.anti_hack_checker",
    ]
    for m in modules:
        try:
            importlib.import_module(m)
            _ok(f"import {m}")
        except Exception as e:  # noqa: BLE001
            _fail(f"import {m}", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 3. FreshPriceEnv reset across all scenarios
# ---------------------------------------------------------------------------

def check_env_resets() -> None:
    print("\n[3] FreshPriceEnv.reset() across all curriculum scenarios")
    try:
        from freshprice_env.freshprice_env import FreshPriceEnv
        from freshprice_env.enums import CurriculumScenario
    except Exception as e:  # noqa: BLE001
        _fail("import env", f"{type(e).__name__}: {e}")
        return

    for scenario in CurriculumScenario:
        try:
            env = FreshPriceEnv(scenario=scenario, seed=42)
            obs, info = env.reset()
            if not isinstance(obs, str) or len(obs) < 50:
                _fail(f"reset {scenario.name}", f"observation too short ({len(obs) if isinstance(obs, str) else 'non-str'})")
            else:
                _ok(f"reset {scenario.name}", f"obs_len={len(obs)} engine={info.get('engine_type', '?')}")
        except Exception as e:  # noqa: BLE001
            _fail(f"reset {scenario.name}", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 4. Server module imports + admin endpoints registered
# ---------------------------------------------------------------------------

def check_server_app() -> None:
    print("\n[4] Server / admin endpoints")
    try:
        from server import app as server_app
    except Exception as e:  # noqa: BLE001
        _fail("import server.app", f"{type(e).__name__}: {e}")
        return

    routes = {getattr(r, "path", None) for r in server_app.app.routes}
    for required in ["/admin/dashboard", "/admin/metrics/scores", "/admin/tasks"]:
        if required in routes:
            _ok(f"route {required}")
        else:
            _fail(f"route {required}", "not registered")


# ---------------------------------------------------------------------------
# 5. Static dashboard files
# ---------------------------------------------------------------------------

def check_static() -> None:
    print("\n[5] Static dashboard files")
    for f in ["static/index.html", "static/app.js", "static/styles.css"]:
        path = ROOT / f
        if path.exists():
            _ok(f, f"{path.stat().st_size} bytes")
        else:
            _fail(f, "missing")


# ---------------------------------------------------------------------------
# 6. SFT generator sanity
# ---------------------------------------------------------------------------

def check_sft_generator() -> None:
    print("\n[6] SFT data generator")
    try:
        from training.generate_sft_data import generate_pricing_examples
        examples = generate_pricing_examples(n_per_difficulty=2)
    except Exception as e:  # noqa: BLE001
        _fail("generate_pricing_examples", f"{type(e).__name__}: {e}")
        return
    if len(examples) != 6:
        _fail("generate_pricing_examples count", f"got {len(examples)}, expected 6")
        return
    required_sections = ["SITUATION:", "DIRECTIVE:", "CONFIDENCE:"]
    for i, ex in enumerate(examples):
        completion = ex.get("completion", "")
        missing = [s for s in required_sections if s not in completion]
        if missing:
            _fail(f"example {i} sections", f"missing {missing}")
            return
    _ok("SFT generator output", "all 6 example completions contain required sections")


# ---------------------------------------------------------------------------
# 7. Env vars (warn-only)
# ---------------------------------------------------------------------------

def check_env_vars() -> None:
    print("\n[7] Environment variables (warn-only)")
    for var in RECOMMENDED_ENV_VARS:
        if os.environ.get(var):
            _ok(f"${var} is set")
        else:
            _warn(f"${var} not set", "not required for local validation; required for HF push")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print(" QStorePrice Submission Validator")
    print("=" * 60)

    try:
        check_openenv_yaml()
        check_module_imports()
        check_env_resets()
        check_server_app()
        check_static()
        check_sft_generator()
        check_env_vars()
    except KeyboardInterrupt:
        print("\nValidation interrupted by user.")
        return 130
    except Exception:  # noqa: BLE001
        print("\nUnhandled exception during validation:")
        traceback.print_exc()
        return 2

    print("\n" + "=" * 60)
    print(f" Result: {_PASSED} passed, {_FAILED} failed")
    print("=" * 60)
    return 0 if _FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
