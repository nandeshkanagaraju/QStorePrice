#!/usr/bin/env python3
"""Run a no-GPU baseline through the real env using RuleBasedAgent.

Writes the same aggregate shape as ``eval/baseline.py`` so
``scripts/compare_baseline_trained.py`` can diff it against a trained eval JSON.

Example:
    python scripts/run_heuristic_baseline_eval.py \\
        --scenarios STABLE_WEEK CRISIS_WEEK --episodes 2 \\
        --output eval/heuristic_baseline_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.baselines.rule_based_agent import RuleBasedAgent
from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def run_heuristic(
    scenarios: list[str],
    episodes_per_scenario: int,
    output_file: str,
) -> dict:
    agent = RuleBasedAgent()
    results: dict[str, dict] = {}

    for scenario_name in scenarios:
        scenario = CurriculumScenario[scenario_name]
        level = scenario.value
        ep_seeds = [level * 1000 + i for i in range(episodes_per_scenario)]

        wrrs: list[float] = []
        qualities: list[float] = []
        violations: list[int] = []

        for i, ep_seed in enumerate(ep_seeds):
            env = FreshPriceEnv(scenario=scenario, seed=ep_seed)
            prompt, info = env.reset(seed=ep_seed)
            done = False
            while not done:
                brief = agent.act(prompt, info)
                prompt, _r, done, _trunc, info = env.step(brief)

            final = info.get("final_reward", {})
            wrrs.append(float(final.get("wrr", 0.0)))
            qualities.append(float(final.get("brief_quality_score", 0.0)))
            violations.append(int(final.get("anti_hack_violations", 0)))
            print(
                f"  {scenario_name} ep {i + 1}/{len(ep_seeds)} "
                f"seed={ep_seed} WRR={wrrs[-1]:.4f} Q={qualities[-1]:.3f} viol={violations[-1]}"
            )

        results[scenario_name] = {
            "wrr_mean": round(_mean(wrrs), 4),
            "wrr_std": round(_std(wrrs), 4),
            "quality_mean": round(_mean(qualities), 4),
            "quality_std": round(_std(qualities), 4),
            "violations_mean": round(_mean([float(v) for v in violations]), 1),
            "episodes": episodes_per_scenario,
        }

    out = {
        "model_id": "heuristic:RuleBasedAgent (no LLM)",
        "evaluation_date": datetime.now().isoformat(),
        "episodes_per_scenario": episodes_per_scenario,
        "results": results,
    }
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {output_file}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Heuristic baseline eval (no torch/unsloth)")
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=["STABLE_WEEK", "CRISIS_WEEK"],
        choices=[s.name for s in CurriculumScenario],
    )
    p.add_argument("--episodes", type=int, default=2, help="Episodes per scenario (seeds level*1000+i)")
    p.add_argument("--output", default="eval/heuristic_baseline_results.json")
    args = p.parse_args()
    print("Heuristic baseline (RuleBasedAgent)")
    print(f"  scenarios: {args.scenarios}")
    print(f"  episodes per scenario: {args.episodes}")
    run_heuristic(args.scenarios, args.episodes, args.output)


if __name__ == "__main__":
    main()
