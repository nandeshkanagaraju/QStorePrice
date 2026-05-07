<div align="center">

# Running QStorePrice AI on Kaggle

### A reproducibility guide for the full Gemma-4 SFT → GRPO rollouts → DPO pipeline

*Submission for **The Gemma 4 Good Hackathon** — Impact Track: Global Resilience · Special Tech Track: Unsloth.*

*End-to-end on a free Kaggle T4 GPU in 45–75 minutes.*

[![Notebook](https://img.shields.io/badge/Notebook-kaggle__qstoreprice.ipynb-20BEFF.svg?logo=kaggle&logoColor=white)](kaggle_qstoreprice.ipynb)
[![Hardware](https://img.shields.io/badge/GPU-Tesla%20T4%20%2F%20P100-76B900.svg?logo=nvidia&logoColor=white)](#2-hardware-requirements)
[![Runtime](https://img.shields.io/badge/Runtime-45--75%20min-blue.svg)](#5-cell-by-cell-walkthrough)
[![Project README](https://img.shields.io/badge/Project-README-lightgrey.svg)](README.md)

**[← Back to README](README.md)** ·
**[Live Demo](https://huggingface.co/spaces/nandeshjeyalakshmi/storeprice-gemma4)** ·
**[Spec](FreshPrice_SDD.md)**

</div>

---

## Table of Contents

| #   | Section                                                              |
| --- | -------------------------------------------------------------------- |
| 1   | [What this notebook trains](#1-what-this-notebook-trains)            |
| 2   | [Hardware requirements](#2-hardware-requirements)                    |
| 3   | [Setup on Kaggle](#3-setup-on-kaggle)                                |
| 4   | [Configuration — Cell 1](#4-configuration--cell-1)                   |
| 5   | [Cell-by-cell walkthrough](#5-cell-by-cell-walkthrough)              |
| 6   | [Outputs](#6-outputs)                                                |
| 7   | [Troubleshooting](#7-troubleshooting)                                |
| 8   | [One-shot reproducibility recipe](#8-one-shot-reproducibility-recipe) |
| 9   | [Beyond Kaggle](#9-beyond-kaggle)                                    |
| 10  | [Where to look next](#10-where-to-look-next)                         |

---

## 1. What this notebook trains

The notebook drives the full RL fine-tuning stack against the FreshPrice Gym
environment described in [`FreshPrice_SDD.md`](FreshPrice_SDD.md):

| Stage                | Framework                  | What it does                                                                                              |
| -------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------- |
| **SFT warm-start**   | Unsloth + TRL `SFTTrainer` | Teaches Gemma 4 the 6-section Operating Brief format from synthetic examples (90–450 per run).            |
| **GRPO rollouts**    | Custom rollout loop        | Runs the SFT model in `FreshPriceEnv` to collect trajectories *(no gradient updates — buffer building).* |
| **DPO fine-tuning**  | TRL `DPOTrainer`           | Builds chosen/rejected pairs from the trajectory buffer and runs the actual RL gradient step.            |
| **Evaluation**       | Greedy decoding            | Deterministic eval on fixed seeds, plus anti-hack pattern scan.                                          |
| **Backend smoke**    | FastAPI                    | Boots the inference server in the background and hits `/health`, `/reset`, `/step`, `/state`.            |
| **HF Hub push**      | `huggingface_hub`          | Optional. Pushes the merged 16-bit checkpoint to your HF account.                                        |

> **Naming clarification.** Despite the name *"GRPO"*, the gradient update used
> is **DPO** — the GRPO cell only collects trajectories. This is documented
> inside the notebook (Section 8).

---

## 2. Hardware requirements

| GPU                     | Status                | Notes                                                              |
| ----------------------- | --------------------- | ------------------------------------------------------------------ |
| **Kaggle T4 (16 GB)**   | ✅ Default target     | Auto-tune picks 1.5B model, batch 1, packing on, 3 GRPO episodes.  |
| **Kaggle P100 (16 GB)** | ✅ Works              | Same as T4 path; slightly slower training throughput.              |
| **Kaggle TPU**          | ❌ Not supported      | Unsloth is CUDA-only.                                              |
| **CPU only**            | ❌ Will OOM / timeout | The smoke tests pass but SFT training is impractically slow.       |

**Wall-clock time on T4 (default config): ≈ 45–75 minutes end-to-end.**

---

## 3. Setup on Kaggle

### 3.1 Create the notebook

1. Sign in to <https://www.kaggle.com>.
2. Click **+ Create → New Notebook**.
3. In the notebook editor, click **File → Import Notebook**.
4. Choose the **GitHub** tab and paste:

   ```text
   https://github.com/nandeshkanagaraju/QStorePrice/blob/main/kaggle_qstoreprice.ipynb
   ```

   *(Or upload the `.ipynb` directly if the GitHub import is blocked.)*

### 3.2 Enable GPU

In the right-hand **Notebook options** panel:

| Setting       | Value                                |
| ------------- | ------------------------------------ |
| Accelerator   | **GPU T4 x1** (or **GPU P100**)      |
| Persistence   | **No persistence** (default is fine) |
| Internet      | **On** (required for pip + HF Hub)   |
| Environment   | **Pin to original environment**      |

> ⚠️ If the GPU dropdown is greyed out, you must **verify your Kaggle
> account by phone** *(Settings → Phone Verification)*. Free GPU is gated
> behind verification.

### 3.3 (Optional) Add your HF token as a Kaggle Secret

Only needed if you want the notebook to push the trained model back to your
HF account. Gemma 4 weights are gated on Hugging Face, so you **do** need a
token (with the Gemma EULA accepted on your HF account) for the initial
model download.

1. Right sidebar → **Add-ons → Secrets**.
2. Click **Add a new secret**.
3. **Label:** `HF_TOKEN`
4. **Value:** a token from <https://huggingface.co/settings/tokens> with
   **Write** scope.
5. Toggle **Attached** to enable it for this notebook.

Then in **Cell 6** of the notebook, uncomment these two lines:

```python
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
```

---

## 4. Configuration — Cell 1

This is the **only cell you ever need to edit**. Everything else is
auto-tuned. Open Cell 1 and review:

```python
# --- Hugging Face ---
HF_TOKEN      = "hf_REPLACE_WITH_YOUR_TOKEN"   # leave placeholder to skip HF push
HF_REPO_ID    = "your-hf-username/qstoreprice-sft"

# --- Model ---
MODEL_ID = "google/gemma-4-e4b-it"             # E4B fits T4; switch to 26B on A100

# --- Auto-tuning master switch ---
AUTO_TUNE = True   # pick epochs/batch/seq-len from detected VRAM + dataset size
```

| Knob                   | Default                              | When to change                                                                  |
| ---------------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| `MODEL_ID`             | `google/gemma-4-e4b-it`              | Switch to `google/gemma-4-26b-it` only with ≥ 40 GB VRAM (Colab A100).          |
| `AUTO_TUNE`            | `True`                               | Set to `False` to use the `*_MANUAL` constants below it instead.                |
| `HF_TOKEN`             | placeholder                          | Replace to enable Hub push, or leave alone to skip the push step.               |
| `HF_REPO_ID`           | `your-hf-username/qstoreprice-sft`   | Replace with `<your-username>/<repo-name>`.                                     |
| `GRPO_EPISODES_MANUAL` | `3`                                  | Bump to `20+` for real reward curves *(adds significant time).*                 |
| `DPO_ENABLED`          | `True`                               | Set `False` to skip the gradient step and ship just the SFT checkpoint.         |
| `SEED`                 | `42`                                 | Change to vary stochastic dataset shuffling and rollout sampling.               |

After saving Cell 1, click **Run All** from the top menu.

---

## 5. Cell-by-cell walkthrough

The notebook has **21 numbered code cells** designed to run top-to-bottom
without manual intervention. Below is what to look for at each step.

### Section 1 — Hardware check

> **Cell 2.** Prints Python version, OS, CUDA availability, GPU name, free
> VRAM.

**Pass criterion:** `cuda_ok = True` and `vram_gb >= 12`. If you see CPU-only,
go back to step 3.2.

### Section 2 — Install dependencies

> **Cells 3–4.** Two-step install.

- **Cell 3** installs Unsloth via its `colab-new` extra. Must run **first**
  because Unsloth reorders `torch` and `transformers` versions.
- **Cell 4** installs the rest: TRL, PEFT, accelerate, bitsandbytes, datasets,
  gymnasium, pydantic, wandb, huggingface_hub, numpy, pandas, scipy, tqdm,
  python-dotenv.

These cells take **5–8 minutes**. Despite Kaggle warning messages, no kernel
restart is required — Unsloth is engineered to work with hot-installed deps.

### Section 3 — Clone repo

> **Cell 5.** Clones `https://github.com/nandeshkanagaraju/QStorePrice` into
> `/kaggle/working/QStorePrice` and adds it to `sys.path`. Re-runs are
> safe — if the directory exists it `git pull`s instead.

### Section 4 — HF authentication

> **Cell 6.** `huggingface_hub.login(token=HF_TOKEN)` if a real token is
> set; otherwise prints "skipping" and continues. Public model downloads
> still work.

### Section 5 — Environment smoke test

> **Cell 7.** Imports `FreshPriceEnv`, calls `env.reset()`, asserts the
> observation string is between 1,000 and 5,000 characters. Catches missing
> modules early.

### Section 5b — Submission validation

> **Cell 7b.** Runs [`validate_submission.py`](validate_submission.py)
> against the cloned repo. Checks:
>
> - All 6 brief sections are produced by the prompt builder.
> - The reward executor returns typed actions for each engine.
> - `/admin/dashboard` is mounted.
> - Static files for the web UI exist.

If this cell prints failures, **do not proceed** — the rest of the notebook
will break in the same way.

### Section 6 — SFT data generation

> **Cells 8a, 8b.** Two-step.

- **Cell 8a** writes JSONL files to `training/sft_data/`. Auto-sized:

  | Model               | Examples per (engine × difficulty) | Total   |
  | ------------------- | ---------------------------------- | ------- |
  | E2B                 | 50                                 | 450     |
  | **E4B (default)**   | **30**                             | **270** |
  | 26B / 31B           | 10                                 | 90      |

- **Cell 8b** loads every file and verifies all 6 required section headers
  appear in each example.

### Section 7 — SFT warm-start

> **Cell 9.** The longest cell — **15–30 minutes on T4**.

Inside the cell:

1. `FastLanguageModel.from_pretrained()` loads Gemma 4 in 4-bit.
2. LoRA adapters added (`r=16, alpha=16, target_modules` = all Q/K/V/O + MLP).
3. TRL `SFTTrainer.train()` runs with the auto-picked epochs/batch/seq-len.
4. A sanity-check generation runs on a fresh prompt; if the output is missing
   any of the 6 section headers, the cell **automatically retrains once**
   with 2× epochs (`SFT_AUTO_RETRIES=1`).
5. Checkpoint saved with `save_pretrained_merged(SFT_DIR, tokenizer,
   save_method="merged_16bit")` — **mandatory** per `CLAUDE.md` (regular
   `save_pretrained` corrupts 4-bit models).

**Expected pass output:** *"All 6 sections present. SFT VERIFIED."*

### Section 8 — GRPO rollouts

> **Cells 10–11.** Trajectory collection (no gradient updates).

- **Cell 10** defines two interchangeable inference backends:
  - `LocalInferenceClient` — uses the just-trained model on GPU.
  - `HFInferenceClient` — calls the HF Inference API *(set
    `USE_HF_INFERENCE_API=True` in Cell 1; faster but consumes HF credits).*
- **Cell 11** runs `GRPO_EPISODES` rollouts in `FreshPriceEnv`. Each episode
  is 672 ticks (7 simulated days × 96 ticks/day at 15-min resolution) with
  84 brief calls per episode. Trajectories are written to
  `trajectory_buffer` for the DPO step.

> **No gradient updates happen here** — this is buffer building only.

### Section 9 — DPO fine-tuning

> **Cell 12.** *The actual RL gradient step.*

1. Frees the SFT model + GRPO inference state from GPU (avoids T4 OOM).
2. Builds preference pairs: top-quartile WRR trajectories = "chosen",
   bottom-quartile = "rejected".
3. `DPOTrainer.train()` runs gradient updates against the SFT reference
   policy (β=0.1, lr=5e-7).
4. Saves the new merged 16-bit checkpoint to `DPO_DIR`.
5. Prints the **pre-DPO vs post-DPO WRR delta** so you can see improvement.

**Auto-skipped if any of these are true:**

- `DPO_ENABLED=False` in Cell 1.
- `VRAM < DPO_MIN_VRAM_GB` (default 12).
- Trajectory buffer has fewer than `DPO_MIN_PAIRS` (default 4) usable pairs.

When skipped, the SFT checkpoint becomes the final model.

### Section 10 — Evaluation

> **Cell 13.** Greedy (`temperature=0`) decoding on three fixed seeds.

Reports:

- Per-episode WRR
- Average reward across the three engines (r1 pricing, r2 farmer, r3 trend)
- Brief format compliance rate
- Anti-hack violation count

### Section 11 — Anti-hack scan

> **Cell 14.** Runs
> [`eval/anti_hack_checker.py`](eval/anti_hack_checker.py) against the top
> 50 trajectories. Detects 8 reward-hacking patterns (early discounts,
> batch acceptance without inspection, etc.).

Empty output here means no exploits were found.

### Section 12 — Reward curves

> **Cell 15.** Plots WRR, r1/r2/r3, and brief quality across episodes from
> `/kaggle/working/episode_log.json`. Skipped gracefully if fewer than 2
> episodes were logged.

### Section 13 — Backend smoke test

> **Cells 16–18.** FastAPI server lifecycle.

| Cell | Purpose                                                                                                       |
| ---: | ------------------------------------------------------------------------------------------------------------- |
|   16 | Starts the FastAPI server on port 8000 in the background.                                                     |
|   17 | Issues HTTP requests to `/health`, `/reset`, `/step`, `/state`. Falls back to a pure-Python sim if unreachable. |
|   18 | Terminates the server cleanly.                                                                                |

### Section 14 — Push to HF Hub

> **Cell 19.** Optional.

If `HF_TOKEN` is real and `HF_REPO_ID` is set, this cell calls
`model.push_to_hub_merged(HF_REPO_ID, tokenizer,
save_method="merged_16bit", token=HF_TOKEN)` — a one-shot upload of the
final checkpoint plus tokenizer.

After it finishes you'll see:

```text
Pushed to https://huggingface.co/<your-username>/qstoreprice-sft
```

### Section 15 — Final summary

> **Cell 20.** Prints the run config, final checkpoint path, eval WRR, and
> HF repo URL.

### Section 16 — Live admin dashboard

> **Cell 21.** Hits `/admin/dashboard` on the local server (or pulls
> metrics directly from the in-process `metrics` store if the server has
> been stopped) and renders the JSON snapshot inline.

Lets judges see WRR, brief quality, anti-hack violations, and engine reward
components without leaving the notebook.

---

## 6. Outputs

After a successful run, the following exist in `/kaggle/working/`:

```text
checkpoints/
├── sft_v1/                  # SFT warm-start (merged 16-bit)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── dpo_round1/              # DPO fine-tuned (merged 16-bit) — only if DPO ran
│   └── ...
└── final/                   # Symlink/copy of whichever is the latest

plots/
├── reward_curve.png         # WRR, r1, r2, r3 over episodes
└── quality_curve.png        # Brief format compliance

episode_log.json             # One entry per evaluation episode
```

To download the final checkpoint from Kaggle:

1. **Notebook → Output** tab.
2. Click `checkpoints/final/` → **Download**.

Or — better — set `HF_TOKEN` / `HF_REPO_ID` in Cell 1 and let Cell 19 push the
merged model to your HF account in one shot.

---

## 7. Troubleshooting

| Symptom                                                                  | Likely cause                                            | Fix                                                                                          |
| ------------------------------------------------------------------------ | ------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `RuntimeError: CUDA out of memory` in Cell 9                             | T4 + sequence too long                                  | Reduce `SFT_MAX_SEQ_LEN_MANUAL` to 1024 and set `AUTO_TUNE=False` in Cell 1.                 |
| `ImportError: cannot import name 'KernelInfo' from 'unsloth_zoo'`        | Stale `unsloth_zoo` cached from a prior session         | Restart kernel and re-run Cell 3 (Unsloth install). Already pinned in current commit.        |
| Cell 9 sanity check fails twice                                          | LoRA didn't converge in given epochs                    | Increase `SFT_EPOCHS_MANUAL` to 8 and `SFT_AUTO_RETRIES` to 2, set `AUTO_TUNE=False`.        |
| `huggingface_hub.utils.GatedRepoError`                                   | Gemma weights are gated and EULA not accepted           | Visit <https://huggingface.co/google/gemma-4-e4b-it>, accept the Gemma terms, then set `HF_TOKEN` in Cell 1. |
| DPO cell prints *"skipping — buffer too small"*                          | GRPO produced < 4 valid trajectories                    | Increase `GRPO_EPISODES_MANUAL` to 6+; check Cell 11 didn't hit token-budget cutoff.         |
| HF push raises `403 Forbidden`                                           | Token has Read scope, not Write                         | Regenerate token at <https://huggingface.co/settings/tokens> with **Write** role.            |
| Server cell hangs                                                        | Port 8000 already used by a previous run                | Restart the kernel and re-run from Cell 16, or change `SERVER_PORT` in Cell 1.               |

---

## 8. One-shot reproducibility recipe

> Hand this 7-step recipe to a judge. They get a fully trained, evaluated,
> and (optionally) HF-pushed model in under 75 minutes.

1. Open [`kaggle_qstoreprice.ipynb`](kaggle_qstoreprice.ipynb) in Kaggle
   *(File → Import → GitHub).*
2. Notebook Options → **GPU T4 x1**, Internet **On**.
3. *(Optional)* Add `HF_TOKEN` as a Kaggle Secret.
4. Edit Cell 1: replace `HF_REPO_ID` with `<your-username>/qstoreprice-sft`.
5. Run All.
6. Wait 45–75 minutes.
7. Read **Cell 20** (final summary) and **Cell 21** (dashboard).

> **Determinism.** `SEED=42` is propagated through `random.Random` instances
> inside the env (per project convention — see `CLAUDE.md`). Greedy eval
> *(Cell 13)* is fully reproducible; SFT/DPO have minor stochastic variation
> from CUDA non-determinism.

---

## 9. Beyond Kaggle

| Target                    | File                                                              | Notes                                                                                  |
| ------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Local CLI (no notebook)   | [`training/train.py`](training/train.py)                          | `python training/train.py --base-model google/gemma-4-26b-it ...`                       |
| HF Space (live demo)      | [`Dockerfile`](Dockerfile) + [`app.py`](app.py)                   | Deployed at <https://huggingface.co/spaces/nandeshjeyalakshmi/storeprice-gemma4>.          |
| Eval-only on a checkpoint | [`eval/evaluator.py`](eval/evaluator.py)                          | `python eval/evaluator.py --checkpoint checkpoints/sft_v1 --episodes 10`                |

---

## 10. Where to look next

| Document                                                                | What's in it                                                                                                |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [`README.md`](README.md)                                                | Project overview, env description, agents, scenarios, results.                                              |
| [`FreshPrice_SDD.md`](FreshPrice_SDD.md)                                | Source of truth for engine behaviour, reward shaping, and the WRR formula.                                  |
| [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)                              | Architecture deep-dive.                                                                                     |
| [`training/sft_trainer.py`](training/sft_trainer.py)                    | SFT internals.                                                                                              |
| [`training/dpo_trainer.py`](training/dpo_trainer.py)                    | DPO internals.                                                                                              |
| [`training/curriculum.py`](training/curriculum.py)                      | 5-level curriculum logic + promotion rule.                                                                  |
| [`eval/anti_hack_checker.py`](eval/anti_hack_checker.py)                | The 8 reward-hacking pattern detectors.                                                                     |
| [`working_output.ipynb`](working_output.ipynb)                          | Full saved outputs from a real Kaggle run — loss curves, eval tables, plots.                                |

---

<div align="center">

**[← Back to README](README.md)** ·
**[Live Demo](https://huggingface.co/spaces/nandeshjeyalakshmi/storeprice-gemma4)** ·
**[Open the Notebook](kaggle_qstoreprice.ipynb)**

<sub>Built for the **Gemma 4 Good Hackathon** with Gemma 4 · Unsloth · TRL · Gymnasium · OpenEnv</sub>

</div>
