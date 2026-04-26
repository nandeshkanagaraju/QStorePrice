#!/usr/bin/env python3
"""Build measured baseline + trained snapshot comparison for README / plots.

1. Runs RuleBasedAgent (heuristic) across all curriculum scenarios — **measured**
   in this repo (no GPU, same seeds as ``eval/baseline.py``: level*1000+i).
2. Merges **trained** WRR from ``eval/fixtures/kaggle_sft_eval_snapshot.json``
   (greedy SFT eval parsed from ``working_output.ipynb``) where present.
3. Writes:
   - ``eval/comparison/baseline_heuristic_measured.json``
   - ``eval/comparison/comparison_summary.json``  (rows for charts + docs)

Then run:
    python scripts/plot_readme_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util

from freshprice_env.enums import CurriculumScenario


def _load_run_heuristic():
    path = _ROOT / "scripts" / "run_heuristic_baseline_eval.py"
    spec = importlib.util.spec_from_file_location("run_heuristic_baseline_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_heuristic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2, help="Episodes per scenario (seeds level*1000+i)")
    p.add_argument(
        "--trained-json",
        default="eval/fixtures/kaggle_sft_eval_snapshot.json",
        help="JSON with results.STABLE_WEEK etc. (trained / SFT eval)",
    )
    p.add_argument(
        "--out-baseline",
        default="eval/comparison/baseline_heuristic_measured.json",
    )
    p.add_argument(
        "--out-summary",
        default="eval/comparison/comparison_summary.json",
    )
    args = p.parse_args()

    run_heuristic = _load_run_heuristic()
    scenarios = [s.name for s in CurriculumScenario]
    print("Measuring heuristic baseline (RuleBasedAgent)…")
    baseline = run_heuristic(scenarios, args.episodes, args.out_baseline)

    trained_path = _ROOT / args.trained_json
    trained_block: dict = {}
    trained_meta: dict = {}
    if trained_path.is_file():
        trained_raw = json.loads(trained_path.read_text(encoding="utf-8"))
        trained_block = trained_raw.get("results") or {}
        trained_meta = {
            "run_label": trained_raw.get("run_label", ""),
            "model_id": trained_raw.get("model_id", ""),
            "source": trained_raw.get("source", ""),
        }
    else:
        print(f"Warning: no trained fixture at {trained_path}")

    rows: list[dict] = []
    for name in scenarios:
        b = baseline["results"].get(name, {})
        t = trained_block.get(name)
        b_wrr = float(b.get("wrr_mean", 0.0))
        t_wrr = float(t["wrr_mean"]) if isinstance(t, dict) and "wrr_mean" in t else None
        delta = round(t_wrr - b_wrr, 4) if t_wrr is not None else None
        rows.append(
            {
                "scenario": name,
                "before_label": "RuleBasedAgent (measured)",
                "before_wrr_mean": b_wrr,
                "before_wrr_std": b.get("wrr_std"),
                "before_quality_mean": b.get("quality_mean"),
                "after_label": trained_meta.get("run_label", "SFT greedy (notebook)"),
                "after_wrr_mean": t_wrr,
                "after_wrr_std": t.get("wrr_std") if isinstance(t, dict) else None,
                "wrr_delta_after_minus_before": delta,
                "trained_available": t_wrr is not None,
            }
        )

    summary = {
        "generated_for": "README section 3 + comparison plot",
        "baseline": {
            "type": baseline["model_id"],
            "evaluation_date": baseline["evaluation_date"],
            "episodes_per_scenario": baseline["episodes_per_scenario"],
            "path": args.out_baseline,
        },
        "trained_fixture": trained_meta | {"path": str(args.trained_json)},
        "scenarios": rows,
    }

    out_sum = _ROOT / args.out_summary
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    out_sum.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out_summary}")
    print("\nNext: python scripts/plot_readme_comparison.py")


if __name__ == "__main__":
    main()
