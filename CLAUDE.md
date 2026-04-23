# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QStorePrice AI is a perishable goods intelligence platform. An RL-trained LLM (Qwen-2.5-7B) writes Operating Briefs that drive three engines: Dynamic Pricing, Farmer Offer, and Social Trend. The unified metric is WRR (Weekly Waste Recovery Rate). Training pipeline: SFT warm-start → GRPO → DPO with 5-level curriculum.

## Source of Truth

`/Users/nandeshnavya/Documents/FreshPrice_SDD.md` is the complete spec. Do not invent business logic not in the spec.

## Build & Run Commands

```bash
# Install
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements_training.txt

# Verify environment
python -c "from freshprice_env.freshprice_env import FreshPriceEnv; env = FreshPriceEnv(); env.reset()"

# Run full training pipeline
python training/train.py --base-model Qwen/Qwen2.5-7B-Instruct --output-dir checkpoints

# Run SFT only
python training/sft_trainer.py --model-id Qwen/Qwen2.5-7B-Instruct --output-dir checkpoints/sft_v1

# Run evaluation
python eval/evaluator.py --checkpoint checkpoints/sft_v1 --episodes 10

# Lint
ruff check .

# Type check
mypy freshprice_env/ training/ eval/
```

## Architecture

- `freshprice_env/` — Gym environment. `FreshPriceEnv.step()` runs 8 simulation ticks per call. LLM writes Operating Briefs (text), `RuleExecutor` converts DIRECTIVE JSON to typed actions, engines simulate outcomes.
- `freshprice_env/engines/` — Three engines (pricing, farmer, trend) each compute their own reward component (r1, r2, r3). `WRRRewardEngine` in `reward.py` combines them.
- `freshprice_env/brief_pipeline/` — prompt_builder → parser → validator → quality_scorer → rule_executor. Parser never raises; returns `ParseResult(success=False)` on failure.
- `training/` — `train.py` orchestrates SFT → GRPO → DPO → curriculum loop. `CurriculumManager` promotes at WRR >= 0.70 over 5 consecutive valid episodes.
- `eval/` — `evaluator.py` runs deterministic (greedy) evaluation. `anti_hack_checker.py` scans for reward hacking patterns.

## Critical Rules

- **Model save**: Always `model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit")`. Never `model.save_pretrained()` after 4-bit training.
- **Reward penalties**: Stored as positive floats in `constants.py`. Applied as negative in engine code (`r1 -= penalty`).
- **Anti-hack guards**: RuleExecutor flags violations (`was_antihack_blocked`). `FreshPriceEnv.step()` reads flags and calls `reward_engine.record_antihack_violation()`. Engines compute rewards. Separation: executor flags, env wires, engines reward.
- **No bare `random`**: All randomness goes through `rng: random.Random` instances for reproducibility.
- **Episode structure**: 672 ticks (7 days × 96 ticks/day at 15-min resolution). Brief fires every 8 ticks (84 briefs/episode).
- **WRR denominator**: Cost added when batch first becomes at-risk (URGENT/CRITICAL). Deduplicated via `_at_risk_seen` set in PricingEngine. Cost stays in denominator even if batch later clears.

## Naming Conventions

Constants use `SCREAMING_SNAKE`: `ANTIHACK_EARLY_DISCOUNT_PRICE_THRESHOLD`, `R1_URGENCY_CLEARANCE_BONUS`.
Entities are frozen dataclasses (except `SimulatedMarketState` which mutates each tick).
Enums are `str, Enum` for JSON serialization except `CurriculumScenario(int, Enum)` for WandB logging.
