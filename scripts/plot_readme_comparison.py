#!/usr/bin/env python3
"""Bar chart: measured heuristic baseline vs trained WRR (where fixture has data).

Reads ``eval/comparison/comparison_summary.json`` (from
``generate_comparison_artifacts.py``) and writes:

- ``eval/plots/readme_wrr_comparison.png``
- ``static/training/readme_wrr_comparison.png`` (for HF Spaces / static hosting)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    summary_path = _ROOT / "eval" / "comparison" / "comparison_summary.json"
    if not summary_path.is_file():
        print(f"Missing {summary_path} — run: python scripts/generate_comparison_artifacts.py")
        sys.exit(1)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = data["scenarios"]
    names = [r["scenario"].replace("_", "\n") for r in rows]
    before = np.array([r["before_wrr_mean"] for r in rows], dtype=float)
    after_list = [r["after_wrr_mean"] if r["trained_available"] else float("nan") for r in rows]
    after = np.array(after_list, dtype=float)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.bar(x - width / 2, before, width, label="Before — RuleBasedAgent (measured)", color="#64748b")
    after_label = "After — SFT greedy (notebook fixture)"
    labeled_after = False
    for xi, av in zip(x, after):
        if np.isfinite(av):
            ax.bar(
                xi + width / 2,
                av,
                width,
                label=after_label if not labeled_after else None,
                color="#059669",
                alpha=0.9,
            )
            labeled_after = True

    ax.set_ylabel("Mean episode WRR")
    ax.set_title("QStorePrice — Measured baseline vs notebook SFT eval (WRR)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    h0, l0 = ax.get_legend_handles_labels()
    ref = ax.axhline(0.7, color="#dc2626", linestyle="--", linewidth=1)
    ax.legend([*h0, ref], [*l0, "Curriculum ref. 0.70 (spec)"], loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_eval = _ROOT / "eval" / "plots" / "readme_wrr_comparison.png"
    out_static = _ROOT / "static" / "training" / "readme_wrr_comparison.png"
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    out_static.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_eval, bbox_inches="tight")
    fig.savefig(out_static, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_eval}")
    print(f"Wrote {out_static}")


if __name__ == "__main__":
    main()
