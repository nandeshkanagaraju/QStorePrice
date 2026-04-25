"""WRR reward-curve logging + plotting for the hackathon "Showing Improvement" metric.

The hackathon scoring weights "Showing Improvement in Rewards" at 20%, with
explicit emphasis on reward curves and before/after evidence. This module
provides:

    EpisodeLogger      — append-only JSONL of per-episode WRR + components.
    plot_reward_curve  — PNG with the GRPO trajectory and three horizontal
                         baselines (zero-shot, post-SFT, post-DPO).
    plot_sft_loss_curve — PNG of SFT training loss from trainer_state.json.

CLI:
    # WRR reward curve
    python eval/reward_curves.py \
        --log-path checkpoints/episode_log.jsonl \
        --baseline-mean 0.05 --sft-mean 0.18 --posttrain-mean 0.74 \
        --output eval/plots/reward_curve.png

    # SFT loss curve
    python eval/reward_curves.py --mode sft-loss \
        --checkpoint-dir checkpoints/sft_v1 \
        --output eval/plots/sft_loss_curve.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class EpisodeLogEntry:
    episode_num: int
    phase: str  # "baseline" | "sft_eval" | "grpo" | "dpo_eval" | "final_eval"
    scenario: str
    curriculum_level: int
    wrr: float
    r1_pricing: float
    r2_farmer: float
    r3_trend: float
    brief_quality_score: float
    anti_hack_violations: int
    constitutional_passed: bool
    episode_valid: bool
    timestamp: str


class EpisodeLogger:
    """Append-only JSONL logger of per-episode WRR.

    Safe to instantiate once at the top of training/train.py and call .log()
    after each episode. Survives Colab disconnects because each line is
    flushed independently.
    """

    def __init__(self, log_path: str | os.PathLike[str]) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        episode_num: int,
        phase: str,
        scenario: str,
        curriculum_level: int,
        result: dict,
    ) -> None:
        """Append one episode result. `result` is the dict returned by trainer.run_episode()."""
        entry = EpisodeLogEntry(
            episode_num=episode_num,
            phase=phase,
            scenario=scenario,
            curriculum_level=curriculum_level,
            wrr=float(result.get("wrr", 0.0)),
            r1_pricing=float(result.get("r1_pricing", 0.0)),
            r2_farmer=float(result.get("r2_farmer", 0.0)),
            r3_trend=float(result.get("r3_trend", 0.0)),
            brief_quality_score=float(result.get("brief_quality_score", 0.0)),
            anti_hack_violations=int(result.get("anti_hack_violations", 0)),
            constitutional_passed=bool(result.get("constitutional_passed", False)),
            episode_valid=bool(result.get("episode_valid", False)),
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )
        with self.log_path.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with self.log_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]


def _rolling_mean(xs: list[float], window: int) -> list[float]:
    if window <= 1 or len(xs) <= 1:
        return list(xs)
    out: list[float] = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_reward_curve(
    log_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    baseline_mean: float | None = None,
    sft_mean: float | None = None,
    posttrain_mean: float | None = None,
    rolling_window: int = 10,
    title: str = "QStorePrice — WRR per Episode (SFT → GRPO → DPO)",
) -> str:
    """Render a PNG showing the GRPO trajectory and before/after baselines.

    The PNG is the artifact judges see. It encodes:
      - per-episode WRR scatter (GRPO + DPO eval points)
      - rolling-mean line so the trend is readable
      - dashed horizontals for zero-shot, post-SFT, post-DPO means
      - vertical bands + labels at curriculum promotions

    Returns the absolute path of the written file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_path = Path(log_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"Episode log not found: {log_path}")

    entries = [json.loads(line) for line in log_path.open() if line.strip()]
    if not entries:
        raise ValueError(f"Episode log is empty: {log_path}")

    grpo_entries = [e for e in entries if e["phase"] == "grpo"]
    dpo_entries = [e for e in entries if e["phase"] == "dpo_eval"]
    final_entries = [e for e in entries if e["phase"] == "final_eval"]

    fig, ax = plt.subplots(figsize=(12, 6))

    if grpo_entries:
        xs = [e["episode_num"] for e in grpo_entries]
        ys = [e["wrr"] for e in grpo_entries]
        ax.scatter(xs, ys, s=14, alpha=0.35, color="#2b7bd1", label="GRPO episode WRR")
        smoothed = _rolling_mean(ys, rolling_window)
        ax.plot(
            xs,
            smoothed,
            color="#2b7bd1",
            linewidth=2,
            label=f"GRPO rolling mean (window={rolling_window})",
        )

    if dpo_entries:
        xs = [e["episode_num"] for e in dpo_entries]
        ys = [e["wrr"] for e in dpo_entries]
        ax.scatter(xs, ys, s=40, marker="^", color="#e07a14", label="DPO eval", zorder=5)

    if final_entries:
        xs = [e["episode_num"] for e in final_entries]
        ys = [e["wrr"] for e in final_entries]
        ax.scatter(xs, ys, s=80, marker="*", color="#c1272d", label="Final eval", zorder=6)

    if baseline_mean is not None:
        ax.axhline(
            baseline_mean,
            linestyle="--",
            color="#888888",
            label=f"Zero-shot mean ({baseline_mean:.3f})",
        )
    if sft_mean is not None:
        ax.axhline(
            sft_mean,
            linestyle="--",
            color="#5aaa5a",
            label=f"Post-SFT mean ({sft_mean:.3f})",
        )
    if posttrain_mean is not None:
        ax.axhline(
            posttrain_mean,
            linestyle="--",
            color="#c1272d",
            label=f"Post-DPO mean ({posttrain_mean:.3f})",
        )

    last_level = None
    for e in entries:
        if e["phase"] != "grpo":
            continue
        if last_level is not None and e["curriculum_level"] != last_level:
            ax.axvline(e["episode_num"], color="#cccccc", linestyle=":", linewidth=1)
            ax.text(
                e["episode_num"],
                ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.95,
                f" L{e['curriculum_level']}: {e['scenario'][:8]}",
                fontsize=8,
                color="#666666",
            )
        last_level = e["curriculum_level"]

    ax.set_xlabel("Episode #")
    ax.set_ylabel("Weekly Waste Recovery Rate (WRR)")
    ax.set_title(title)
    ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    return str(output_path.resolve())


def plot_sft_loss_curve(
    checkpoint_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    title: str = "QStorePrice — SFT Training Loss",
) -> str:
    """Render a PNG of SFT training loss from a HuggingFace trainer_state.json.

    HF TRL's SFTTrainer writes trainer_state.json in the output directory after
    training (and at each checkpoint). ``log_history`` entries with a ``loss``
    key are training steps; entries with ``eval_loss`` are evaluation steps.

    Returns the absolute path of the written file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoint_dir = Path(checkpoint_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_file = checkpoint_dir / "trainer_state.json"
    if not state_file.exists():
        raise FileNotFoundError(
            f"trainer_state.json not found in {checkpoint_dir}. "
            "Run SFT training first or point --checkpoint-dir at the correct directory."
        )

    with state_file.open() as f:
        state = json.load(f)

    log_history: list[dict] = state.get("log_history", [])
    if not log_history:
        raise ValueError(f"log_history is empty in {state_file}")

    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_loss = [e["loss"] for e in log_history if "loss" in e]
    eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    fig, ax = plt.subplots(figsize=(10, 5))

    if train_steps:
        ax.plot(
            train_steps,
            train_loss,
            color="#2b7bd1",
            linewidth=1.5,
            label="Training loss",
        )
        ax.scatter(train_steps, train_loss, s=10, color="#2b7bd1", alpha=0.4)

    if eval_steps:
        ax.plot(
            eval_steps,
            eval_loss,
            color="#e07a14",
            linewidth=2,
            linestyle="--",
            label="Eval loss",
            zorder=5,
        )
        ax.scatter(eval_steps, eval_loss, s=30, color="#e07a14", zorder=6)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    return str(output_path.resolve())


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reward curves or SFT loss from training logs."
    )
    parser.add_argument(
        "--mode",
        choices=["reward", "sft-loss"],
        default="reward",
        help="reward: WRR curve from episode_log.jsonl; sft-loss: training loss from trainer_state.json",
    )
    parser.add_argument("--output", required=True, help="Output PNG path")

    # reward-mode args
    parser.add_argument("--log-path", help="Path to episode_log.jsonl (--mode reward)")
    parser.add_argument("--baseline-mean", type=float, default=None)
    parser.add_argument("--sft-mean", type=float, default=None)
    parser.add_argument("--posttrain-mean", type=float, default=None)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument(
        "--title", default="QStorePrice — WRR per Episode (SFT → GRPO → DPO)"
    )

    # sft-loss-mode args
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory containing trainer_state.json (--mode sft-loss)",
    )
    parser.add_argument(
        "--loss-title", default="QStorePrice — SFT Training Loss"
    )

    args = parser.parse_args()

    if args.mode == "sft-loss":
        if not args.checkpoint_dir:
            parser.error("--checkpoint-dir is required for --mode sft-loss")
        out = plot_sft_loss_curve(
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output,
            title=args.loss_title,
        )
        print(f"Wrote SFT loss curve PNG: {out}")
    else:
        if not args.log_path:
            parser.error("--log-path is required for --mode reward")
        out = plot_reward_curve(
            log_path=args.log_path,
            output_path=args.output,
            baseline_mean=args.baseline_mean,
            sft_mean=args.sft_mean,
            posttrain_mean=args.posttrain_mean,
            rolling_window=args.rolling_window,
            title=args.title,
        )
        print(f"Wrote reward curve PNG: {out}")


if __name__ == "__main__":
    _cli()
