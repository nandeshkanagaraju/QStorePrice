"""Extract PNG plots and a small metrics snapshot from working_output.ipynb.

Writes:
  static/training/training_metrics.png
  static/training/eval_wrr_by_scenario.png
  static/training/training_snapshot.json

Re-run after regenerating the notebook:
  python scripts/export_notebook_training_assets.py
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "working_output.ipynb"
OUT_DIR = ROOT / "static" / "training"


def _cell_texts(cell: dict) -> str:
    parts: list[str] = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append("".join(o.get("text", [])))
        elif o.get("output_type") in ("display_data", "execute_result"):
            d = o.get("data", {})
            tp = d.get("text/plain")
            if isinstance(tp, list):
                tp = "".join(tp)
            if tp:
                parts.append(str(tp))
    return "\n".join(parts)


def _extract_pngs(cell: dict) -> list[bytes]:
    blobs: list[bytes] = []
    for o in cell.get("outputs", []):
        if o.get("output_type") not in ("display_data", "execute_result"):
            continue
        d = o.get("data", {})
        b64 = d.get("image/png")
        if isinstance(b64, str):
            blobs.append(base64.b64decode(b64))
        elif isinstance(b64, list):
            blobs.append(base64.b64decode("".join(b64)))
    return blobs


def main() -> int:
    if not NB.exists():
        print(f"Missing {NB}")
        return 1
    nb = json.loads(NB.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combined_text = ""
    png_order: list[bytes] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        combined_text += "\n" + _cell_texts(cell)
        png_order.extend(_extract_pngs(cell))

    # Name by typical notebook order (cell 15 plots): 4-panel then scenario chart
    names = ["training_metrics.png", "eval_wrr_by_scenario.png"]
    for i, blob in enumerate(png_order[: len(names)]):
        (OUT_DIR / names[i]).write_bytes(blob)
    if not png_order:
        print("No image/png outputs found in notebook.")

    loss_m = re.search(r"Training loss\s*:\s*([0-9.]+)", combined_text)
    runtime_m = re.search(r"Runtime\s*:\s*([0-9.]+)s", combined_text)
    model_m = re.search(r"MODEL_ID\s*:\s*(\S+)", combined_text)

    grpo_rows: list[dict[str, float | int | str]] = []
    for line in combined_text.splitlines():
        # e.g. "   1  1.991  0.036  0.000  0.000  0.764     5   PASS   943s"
        m = re.match(
            r"\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+(\w+)",
            line,
        )
        if m:
            grpo_rows.append(
                {
                    "episode": int(m.group(1)),
                    "col2": float(m.group(2)),
                    "col3": float(m.group(3)),
                    "col4": float(m.group(4)),
                    "col5": float(m.group(5)),
                    "col6": float(m.group(6)),
                    "violations": int(m.group(7)),
                    "pass": m.group(8),
                }
            )

    wrr_block = re.search(
        r"WRR\s*:\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", combined_text, re.I
    )

    snapshot = {
        "source_notebook": str(NB.name),
        "model_id": model_m.group(1) if model_m else None,
        "sft_training_loss": float(loss_m.group(1)) if loss_m else None,
        "sft_runtime_seconds": float(runtime_m.group(1)) if runtime_m else None,
        "grpo_rollout_rows": grpo_rows[:12],
        "eval_wrr_summary": wrr_block.group(0) if wrr_block else None,
        "plots": names[: len(png_order)],
        "dpo_note": "DPO skipped" if "DPO skipped" in combined_text else None,
    }
    (OUT_DIR / "training_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    print("Wrote:", OUT_DIR / "training_snapshot.json")
    for n in names[: len(png_order)]:
        print("Wrote:", OUT_DIR / n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
