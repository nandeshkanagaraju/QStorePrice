#!/usr/bin/env python3
"""Compare two evaluation JSON files (baseline vs trained / SFT / RL export).

Supported shapes:
  * Output of ``eval/baseline.py`` or ``scripts/run_heuristic_baseline_eval.py``:
    ``{ "results": { "STABLE_WEEK": { "wrr_mean", "wrr_std", "quality_mean", ... }}}``
  * Evaluator-style summary export:
    ``{ "summary": { "by_scenario": { ... }, "overall_wrr_mean": ... }}}``
  * Same as baseline but arbitrary top-level ``label`` / ``run_label`` for printing.

Example:
    python scripts/compare_baseline_trained.py \\
        --baseline eval/heuristic_baseline_results.json \\
        --trained eval/fixtures/kaggle_sft_eval_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_rows(data: dict[str, Any]) -> tuple[dict[str, dict[str, float]], str]:
    """Return (scenario_name -> stats, human label)."""
    label = str(
        data.get("run_label")
        or data.get("label")
        or data.get("model_id")
        or data.get("checkpoint_dir")
        or "run",
    )
    if "results" in data and isinstance(data["results"], dict):
        rows: dict[str, dict[str, float]] = {}
        for name, block in data["results"].items():
            if not isinstance(block, dict):
                continue
            rows[name] = {
                "wrr_mean": float(block.get("wrr_mean", 0.0)),
                "wrr_std": float(block.get("wrr_std", 0.0)),
                "quality_mean": float(block.get("quality_mean", 0.0)),
                "violations_mean": float(block.get("violations_mean", 0.0)),
            }
        return rows, label

    summary = data.get("summary") or {}
    by_scenario = summary.get("by_scenario") or {}
    if isinstance(by_scenario, dict) and by_scenario:
        rows = {}
        for name, block in by_scenario.items():
            if not isinstance(block, dict):
                continue
            rows[name] = {
                "wrr_mean": float(block.get("wrr_mean", 0.0)),
                "wrr_std": float(block.get("wrr_std", 0.0)),
                "quality_mean": float(block.get("quality_mean", 0.0)),
                "violations_mean": float(block.get("violations_mean", 0.0)),
            }
        return rows, label

    raise ValueError(
        f"Unrecognized JSON shape in {label!r}: need top-level 'results' "
        "or 'summary.by_scenario'."
    )


def _overall(rows: dict[str, dict[str, float]]) -> float:
    if not rows:
        return 0.0
    return round(sum(r["wrr_mean"] for r in rows.values()) / len(rows), 4)


def compare(a_path: str, b_path: str, name_a: str, name_b: str) -> None:
    da, db = _load(a_path), _load(b_path)
    rows_a, la = _extract_rows(da)
    rows_b, lb = _extract_rows(db)

    scenarios = sorted(set(rows_a) | set(rows_b))

    print()
    print("=" * 72)
    print("BASELINE vs TRAINED — WRR / quality / violations (per scenario)")
    print("=" * 72)
    print(f"  A ({name_a}): {la}")
    print(f"     file: {a_path}")
    print(f"  B ({name_b}): {lb}")
    print(f"     file: {b_path}")
    print()
    hdr = f"  {'Scenario':<18} {'WRR A':>10} {'WRR B':>10} {'Δ WRR':>10} {'Qual A':>8} {'Qual B':>8} {'Viol A':>8} {'Viol B':>8}"
    print(hdr)
    print(f"  {'-' * 68}")

    deltas: list[float] = []
    for s in scenarios:
        ra = rows_a.get(s, {"wrr_mean": float("nan"), "quality_mean": float("nan"), "violations_mean": float("nan")})
        rb = rows_b.get(s, {"wrr_mean": float("nan"), "quality_mean": float("nan"), "violations_mean": float("nan")})
        if math.isnan(ra["wrr_mean"]) or math.isnan(rb["wrr_mean"]):
            dw = float("nan")
        else:
            dw = round(rb["wrr_mean"] - ra["wrr_mean"], 4)
            deltas.append(dw)
        print(
            f"  {s:<18} {ra['wrr_mean']:>10.4f} {rb['wrr_mean']:>10.4f} {dw:>+10.4f} "
            f"{ra['quality_mean']:>8.3f} {rb['quality_mean']:>8.3f} "
            f"{ra['violations_mean']:>8.1f} {rb['violations_mean']:>8.1f}"
        )

    oa, ob = _overall(rows_a), _overall(rows_b)
    print(f"  {'-' * 68}")
    print(f"  {'MEAN(scenarios)':<18} {oa:>10.4f} {ob:>10.4f} {round(ob - oa, 4):>+10.4f}")
    print()
    if deltas and all(d > 0 for d in deltas):
        print("  Summary: B is higher on every overlapping scenario (by mean WRR).")
    elif deltas and all(d < 0 for d in deltas):
        print("  Summary: A is higher on every overlapping scenario (by mean WRR).")
    else:
        print("  Summary: Mixed — compare per scenario above.")
    print("=" * 72)
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Compare baseline vs trained eval JSON")
    p.add_argument("--baseline", required=True, help="JSON from baseline.py or heuristic script")
    p.add_argument("--trained", required=True, help="JSON from trained eval / fixture / export")
    p.add_argument("--label-a", default="A / baseline")
    p.add_argument("--label-b", default="B / trained")
    args = p.parse_args()
    compare(args.baseline, args.trained, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
