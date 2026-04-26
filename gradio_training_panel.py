"""Gradio helpers: training snapshot + environment / scenario / agent copy."""

from __future__ import annotations

import json
from pathlib import Path

from freshprice_env.enums import CurriculumScenario

_TRAIN_DIR = Path(__file__).resolve().parent / "static" / "training"
_SNAPSHOT = _TRAIN_DIR / "training_snapshot.json"


def load_training_snapshot() -> dict:
    if _SNAPSHOT.exists():
        return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    return {}


def environment_markdown() -> str:
    return """### FreshPrice / QStorePrice environment

| Property | Value |
|----------|------|
| Episode length | **672** ticks (7 days × 96 ticks/day @ 15 min) |
| Briefs per episode | **84** (one brief every **8** ticks) |
| Unified metric | **WRR** — Weekly Waste Recovery Rate |
| Action | Structured **Operating Brief** (6 sections + `DIRECTIVE` JSON) |
| Anti-hack | Rule executor flags pathological strategies; env applies penalties |

The gym-style core is `FreshPriceEnv`; the OpenEnv HTTP adapter is `FreshPriceOpenEnv` for Spaces and clients.
"""


def scenarios_markdown() -> str:
    lines = [
        "### Curriculum scenarios (agents face these worlds)",
        "",
        "| Scenario | Level | Active engines | Notes |",
        "|----------|-------|----------------|-------|",
    ]
    engines = {
        CurriculumScenario.STABLE_WEEK: "Pricing",
        CurriculumScenario.BUSY_WEEKEND: "Pricing, Trend",
        CurriculumScenario.FARMER_WEEK: "Pricing, Farmer",
        CurriculumScenario.TREND_WEEK: "Pricing, Farmer, Trend",
        CurriculumScenario.CRISIS_WEEK: "Pricing, Farmer, Trend",
    }
    blurb = {
        CurriculumScenario.STABLE_WEEK: "Predictable week.",
        CurriculumScenario.BUSY_WEEKEND: "Weekend surge + 1 trend.",
        CurriculumScenario.FARMER_WEEK: "3 farmer offers, no trends.",
        CurriculumScenario.TREND_WEEK: "2 trends + festival spike.",
        CurriculumScenario.CRISIS_WEEK: "Benchmark: all stressors on.",
    }
    for s in CurriculumScenario:
        lines.append(
            f"| **{s.name}** | {int(s.value)} | {engines[s]} | {blurb[s]} |"
        )
    lines.append("")
    lines.append("Promotion (levels 0–3): **WRR ≥ 0.70** on **5** consecutive valid episodes.")
    return "\n".join(lines)


def agents_markdown() -> str:
    return """### Agents (who does what)

1. **LLM policy (trainable)** — Writes the **Operating Brief**: situation, analysis, viability, recommendation, machine-readable `DIRECTIVE`, and confidence. Trained with **SFT → rollouts → DPO** (see *Training* tab for the latest exported run).
2. **Rule executor (deterministic)** — Parses `DIRECTIVE` JSON and applies typed actions; never “invents” economics.
3. **Simulation engines (environment)**  
   - **Dynamic pricing** — discounts vs expiry urgency (**r1**)  
   - **Farmer offers** — accept / counter / decline (**r2**)  
   - **Social trend** — restock vs viral demand (**r3**)  

Together they produce **WRR** and per-brief component rewards used for learning and dashboards.
"""


def training_run_markdown(snap: dict) -> str:
    if not snap:
        return (
            "### Training snapshot\n\nNo `static/training/training_snapshot.json` yet. "
            "Run:\n\n```bash\npython scripts/export_notebook_training_assets.py\n```\n\n"
            "after updating `working_output.ipynb`."
        )
    lines = [
        "### Latest export from `working_output.ipynb`",
        "",
        f"- **Model:** `{snap.get('model_id', '—')}`",
        f"- **SFT final loss:** `{snap.get('sft_training_loss', '—')}`",
        f"- **SFT wall time (s):** `{snap.get('sft_runtime_seconds', '—')}`",
        f"- **Eval (from logs):** `{snap.get('eval_wrr_summary', '—')}`",
    ]
    if snap.get("dpo_note"):
        lines.append(f"- **DPO:** _{snap['dpo_note']}_")
    rows = snap.get("grpo_rollout_rows") or []
    if rows:
        lines.extend(["", "#### GRPO rollout table (parsed from notebook logs)", ""])
        lines.append("| Ep | c2 | c3 | c4 | c5 | c6 | viol | pass |")
        lines.append("|----|----|----|----|----|----|------|------|")
        for r in rows:
            lines.append(
                f"| {r.get('episode')} | {r.get('col2')} | {r.get('col3')} | "
                f"{r.get('col4')} | {r.get('col5')} | {r.get('col6')} | "
                f"{r.get('violations')} | {r.get('pass')} |"
            )
    lines.append("")
    lines.append("Plots below are the same PNGs saved from the notebook cells.")
    return "\n".join(lines)


def training_plot_paths() -> tuple[str | None, str | None]:
    a, b = _TRAIN_DIR / "training_metrics.png", _TRAIN_DIR / "eval_wrr_by_scenario.png"
    return (str(a) if a.exists() else None, str(b) if b.exists() else None)
