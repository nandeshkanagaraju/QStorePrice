---
title: QStorePrice
emoji: 🥭
colorFrom: green
colorTo: orange
sdk: docker
app_file: app.py
pinned: false
---

# QStorePrice AI — Perishable Goods Intelligence

> Can an RL-trained LLM manage every pricing, procurement, and restocking
> decision a small grocery dark store faces — better than gut instinct alone?

---

## 1. The Problem

Indian grocery dark stores lose **15–30% of perishable inventory** to expiry.
The decisions that drive this waste — when to discount, whether to accept a
farmer's surplus offer, and whether a viral food trend justifies a restock —
happen under time pressure with no decision support. Human buyers are fast but
inconsistent: the same mango surplus gets accepted on Tuesday and rejected on
Thursday for no documented reason.

Standard numeric-action RL fails here because the action space is not a
price multiplier float — it is a *reasoning chain* that must weigh shelf life,
cash buffer, demand timing, and competitive signals simultaneously. A model
that produces a correct number for the wrong reason is not useful.

**QStorePrice AI closes this gap**: it trains a 7B LLM via RL (SFT + GRPO + DPO)
to write structured **Operating Briefs** that make the reasoning visible,
auditable, and measurable — not just the outcome.

---

## 2. The Environment

### What the agent sees

Every 2 simulated hours (8 ticks at 15-min resolution) the agent receives a
structured prompt describing:

- Active inventory batches with expiry urgency (`WATCH / URGENT / CRITICAL`)
- Open farmer surplus offers with viability pre-computation
- Active social trend signals with projected demand lift
- Current cash risk buffer and WRR accumulator

### What the agent does

The agent writes an **Operating Brief** — a 6-section structured document:

```
SITUATION:     Farmer Rajan offers 50 kg mangoes at Rs 35/kg, 48 hrs shelf life.
               Current mango inventory: 12 units at WATCH (35 hrs remaining).

SIGNAL ANALYSIS: No active trend signals for mangoes.

VIABILITY CHECK:
  Shelf life: PASS — 48 hrs covers projected sell-through of 32 hrs
  Break-even: PASS — market Rs 75/kg vs break-even Rs 43/kg (74% margin)
  Worst-case P&L: FLAG — 60% sell-through at Rs 47/kg barely covers cost

RECOMMENDATION: ACCEPT. Strong viability (0.78) with healthy buffer.

DIRECTIVE: {"engine": "FARMER", "actions": [{"offer_id": "offer_001", "decision": "ACCEPT"}]}

CONFIDENCE: HIGH
```

A deterministic **Rule Executor** converts the `DIRECTIVE` JSON into typed
actions for three engines. The LLM does language work; the executor does math.

### What the agent is rewarded for

All three engines collapse into one metric:

> **WRR (Weekly Waste Recovery Rate)** = revenue recovered from at-risk inventory
> / cost of at-risk inventory

| Engine | Reward Component | Weight |
|--------|-----------------|--------|
| Dynamic Pricing — auto-discount items nearing expiry | `r1_pricing` | 0.50 |
| Farmer Offer — accept / counter / decline surplus | `r2_farmer` | 0.30 |
| Social Trend — restock before viral demand arrives | `r3_trend` | 0.20 |

**Target WRR**: 0.61 (baseline gut-instinct) → 0.89 (trained model)

### Training pipeline

```
SFT warm-start  →  GRPO (5-level curriculum)  →  DPO (preference pairs)
   50 briefs          STABLE → BUSY → FARMER        WRR-filtered
                      → TREND → CRISIS              + anti-hack audit
```

Curriculum promotes at WRR ≥ 0.70 over 5 consecutive valid episodes.
CRISIS_WEEK (Level 4) is the benchmark: simultaneous supplier delay,
viral trend, 3 farmer offers, and depleted risk buffer.

### Anti-hack guards

The environment blocks pathological strategies that game WRR without genuine
decision quality:

| Guard | Trigger | Consequence |
|-------|---------|-------------|
| Early deep discount | `price_multiplier < 0.35` with `hours_to_expiry > 48` | Price not applied; r1 penalty |
| Reckless acceptance | ACCEPT with `viability_score < 0.30` | Forced DECLINE; r2 penalty |
| Trend order flood | > 1 order per category within 72 hrs | Hard cap; r3 penalty |
| Surrogate gaming | `brief_quality > 0.90` but `WRR < 0.50` | Excluded from DPO pairs |

---

## 3. Results

*Results below are projected from curriculum promotion thresholds.
Run training and evaluation to replace with actuals.*

### WRR: Before vs After Training

| Scenario | Zero-Shot WRR | Trained WRR | Improvement |
|----------|--------------|-------------|-------------|
| STABLE_WEEK | ~0.09 | ~0.72 | +685% |
| BUSY_WEEKEND | ~0.06 | ~0.70 | +1067% |
| FARMER_WEEK | ~0.04 | ~0.68 | +1485% |
| TREND_WEEK | ~0.07 | ~0.65 | +857% |
| CRISIS_WEEK | ~0.03 | ~0.58 | +1833% |

### Reasoning quality vs WRR correlation

The research question: does **brief quality** improve alongside WRR, or does
the model find WRR shortcuts that bypass reasoning?

`brief_quality_score` is scored independently of WRR (structural completeness +
claim grounding + confidence calibration). If it *leads* WRR improvement by a
few episodes during GRPO, the model is learning to reason better — not gaming.

*Correlation plot — generate after training by running `eval/evaluator.py`
and `eval/reward_curves.py`. The PNG will appear in `eval/plots/`.*

### Training curves

*Reward curve plots — generated automatically during training and saved to
`eval/plots/` by `eval/reward_curves.py`.*

---

## 4. Why It Matters

**For the hackathon**: QStorePrice demonstrates that an LLM writing structured
operating documents is a viable RL agent. The brief format makes the reward
signal interpretable (you can read *why* the model got a high r2 score this
episode), prevents reward hacking through constitutional checks, and produces
artefacts (the briefs themselves) that are useful outside the training loop.

**For the domain**: A trained QStorePrice model is a drop-in decision assistant
for any quick-commerce operator running perishable SKUs. The Operating Brief
format can be audited by a human buyer, logged for compliance, and continuously
improved by running more GRPO episodes on new scenarios.

**For RL research**: The `brief_quality_score` / `WRR` correlation across
training episodes is a falsifiable claim about whether RL improves LLM reasoning
or just WRR-gaming. One model family (Qwen-2.5-7B), one reward function — but
a replicable methodology.

---

## Project Structure

```
freshprice_env/          # Gym-style simulation environment
  freshprice_env.py      # FreshPriceEnv(gym.Env) — 672-tick episodes
  engines/               # pricing_engine, farmer_engine, trend_engine
  brief_pipeline/        # prompt_builder → parser → validator → rule_executor
  reward.py              # WRRRewardEngine (r1 + r2 + r3 → WRR)
  openenv_adapter.py     # OpenEnv wrapper (BriefAction, BriefObservation, FreshPriceState)

models.py                # Top-level re-exports for pip clients
client.py                # QStorePriceEnv(HTTPEnvClient) — remote client
server/app.py            # FastAPI server via create_fastapi_app
openenv.yaml             # Environment manifest
Dockerfile               # HF Space / Docker image

training/
  train.py               # SFT → GRPO → DPO orchestration
  sft_trainer.py         # SFT warm-start
  grpo_trainer.py        # GRPO training loop
  dpo_trainer.py         # DPO fine-tuning
  curriculum.py          # 5-level curriculum manager

eval/
  evaluator.py           # Deterministic evaluation (greedy decoding)
  reward_curves.py       # WRR + loss curve plots → eval/plots/
  anti_hack_checker.py   # Trajectory anti-hack audit
```

## Quick Start

```bash
# Install (GPU required for training)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements_training.txt

# Verify environment (CPU-only, no GPU needed)
python -c "from freshprice_env.freshprice_env import FreshPriceEnv; env = FreshPriceEnv(); env.reset()"

# Run training pipeline
python training/train.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --output-dir checkpoints \
  --wandb-project qstoreprice-ai

# Evaluate a checkpoint
python eval/evaluator.py \
  --checkpoint checkpoints/promoted_level1 \
  --episodes 10
```

## OpenEnv Server

```bash
# Start the OpenEnv FastAPI server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Or via Docker
docker run -d -p 8000:8000 <image>

# Connect with the client
python -c "
from client import QStorePriceEnv
env = QStorePriceEnv('http://localhost:8000')
obs, info = env.reset()
print(obs[:200])
"
```

Endpoints: `GET /health` · `POST /reset` · `POST /step` · `GET /state` · `WS /ws` · `GET /docs`

## Training Scenarios

| Level | Scenario | Engines Active | Promotion |
|-------|----------|----------------|-----------|
| 0 | STABLE_WEEK | Pricing only | WRR ≥ 0.70 × 5 episodes |
| 1 | BUSY_WEEKEND | Pricing + Trend | WRR ≥ 0.70 × 5 episodes |
| 2 | FARMER_WEEK | Pricing + Farmer | WRR ≥ 0.70 × 5 episodes |
| 3 | TREND_WEEK | All 3 engines | WRR ≥ 0.70 × 5 episodes |
| 4 | CRISIS_WEEK | All 3 engines | Benchmark only |

## Links

- **HuggingFace Space**: ADD_AFTER_TRAINING — `openenv push` will print the URL
- **WandB Training Run**: ADD_AFTER_TRAINING — set `--wandb-project qstoreprice-ai`
- **OpenEnv version**: openenv-core ≥ 0.2.0 (hackathon target: v0.2.3)
- **Base model**: Qwen/Qwen2.5-7B-Instruct
- **Environment class**: `freshprice_env.openenv_adapter.FreshPriceOpenEnv`
- **Manifest**: `openenv.yaml`

## Citation

```bibtex
@software{qstoreprice_ai_2026,
  title   = {QStorePrice AI: RL-Trained LLM for Perishable Goods Intelligence},
  year    = {2026},
  url     = {https://github.com/nandeshkanagaraju/QStorePrice},
}
```
