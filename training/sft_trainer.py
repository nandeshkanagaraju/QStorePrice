"""SFT warm-start trainer — teaches the model Operating Brief format and basic decision logic.

Runs before GRPO so the model produces parseable briefs from episode 1.
Uses Unsloth 4-bit quantisation and HuggingFace TRL SFTTrainer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset

from freshprice_env.brief_pipeline.prompt_builder import OperatingBriefPromptBuilder

logger = logging.getLogger(__name__)

# Chat template tokens for Gemma-4 instruct models.
# Gemma-4 has no native system role; the system prompt is folded into the
# first user turn (the same approach Gemma's tokenizer.apply_chat_template
# uses). The tags below are used to assemble training-time strings; for
# inference we recommend `tokenizer.apply_chat_template(...)` directly.
_SYSTEM_TAG = "<start_of_turn>user"  # system content is prepended to user turn
_USER_TAG = "<start_of_turn>user"
_ASSISTANT_TAG = "<start_of_turn>model"
_END_TAG = "<end_of_turn>"

# Required section headers that must appear in a valid brief
_REQUIRED_SECTIONS = ["SITUATION:", "SIGNAL ANALYSIS:", "VIABILITY CHECK:",
                      "RECOMMENDATION:", "DIRECTIVE:", "CONFIDENCE:"]

# Fixed test prompt for checkpoint verification
_VERIFICATION_PROMPT = (
    "=== CURRENT INVENTORY ===\n"
    "[🔴] dairy — Batch batch_test_001\n"
    "  Stock: 12 units | Expiry: 3.5hrs | Urgency: CRITICAL\n"
    "  Current price: Rs 75 | Original: Rs 75 | Floor: Rs 32\n"
    "  Velocity: 1.20 units/hr | Discount: 0%\n"
    "\n"
    "[🟢] fruits — Batch batch_test_002\n"
    "  Stock: 20 units | Expiry: 85.0hrs | Urgency: FRESH\n"
    "  Current price: Rs 60 | Original: Rs 60 | Floor: Rs 25\n"
    "  Velocity: 2.50 units/hr | Discount: 0%\n"
    "\n"
    "=== MARKET CONTEXT ===\n"
    "Day: Wednesday | Hour: 2:00 PM | Risk Buffer: Rs 5000\n"
    "Notification credits remaining: dairy: 3, fruits: 3\n"
    "\n"
    "=== YOUR TASK ===\n"
    "Write a PRICING Operating Brief. For each URGENT or CRITICAL batch, "
    "decide the appropriate price_multiplier. For FRESH and WATCH batches, "
    "you may hold price (1.0) or apply a small early discount.\n"
    "Remember: price_multiplier below 0.35 on batches with > 48hrs remaining "
    "triggers an anti-hack penalty."
)


def load_sft_dataset(data_dir: str = "training/sft_data") -> Dataset:
    """Load all three SFT JSON files and merge into a HuggingFace Dataset.

    Each example is formatted as a chat-style text string for SFTTrainer.
    Shuffled with fixed seed 42 after merging.
    """
    data_path = Path(data_dir)
    all_examples: list[dict] = []

    system_prompt = OperatingBriefPromptBuilder.SYSTEM_PROMPT

    for json_file in sorted(data_path.glob("*.json")):
        with open(json_file) as f:
            examples = json.load(f)
        logger.info("Loaded %d examples from %s", len(examples), json_file.name)
        all_examples.extend(examples)

    if not all_examples:
        raise FileNotFoundError(f"No SFT data found in {data_dir}")

    # Format for SFTTrainer.
    # Gemma 4 has no dedicated system role — the system prompt is folded into
    # the first user turn so the model sees a single, well-formed conversation.
    formatted: list[dict] = []
    for ex in all_examples:
        text = (
            f"{_USER_TAG}\n{system_prompt}\n\n{ex['prompt']}{_END_TAG}\n"
            f"{_ASSISTANT_TAG}\n{ex['completion']}{_END_TAG}"
        )
        formatted.append({
            "text": text,
            "engine_type": ex.get("engine_type", "UNKNOWN"),
            "difficulty": ex.get("difficulty", "unknown"),
        })

    dataset = Dataset.from_list(formatted)
    dataset = dataset.shuffle(seed=42)

    # Print statistics
    engine_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for ex in formatted:
        et = ex["engine_type"]
        engine_counts[et] = engine_counts.get(et, 0) + 1
        diff = ex["difficulty"]
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    logger.info("SFT dataset: %d total examples", len(formatted))
    logger.info("  By engine: %s", engine_counts)
    logger.info("  By difficulty: %s", difficulty_counts)
    print(f"SFT dataset loaded: {len(formatted)} examples")
    print(f"  By engine: {engine_counts}")
    print(f"  By difficulty: {difficulty_counts}")

    return dataset


def run_sft(
    model_id: str = "google/gemma-4-26b-it",
    output_dir: str = "checkpoints/sft_v1",
    data_dir: str = "training/sft_data",
    num_epochs: int = 2,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 4096,
    seed: int = 42,
) -> str:
    """Run SFT warm-start training. Returns path to saved checkpoint."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    # 1. Load model and tokenizer with Unsloth 4-bit
    print(f"Loading model: {model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # 3. Load dataset
    dataset = load_sft_dataset(data_dir)

    # 4. Configure SFTTrainer
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=50,
        seed=seed,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # 5. Train
    print("Starting SFT training...")
    trainer_stats = trainer.train()
    print(f"Training complete.")
    print(f"  Loss: {trainer_stats.training_loss:.4f}")
    print(f"  Runtime: {trainer_stats.metrics.get('train_runtime', 0):.1f}s")
    print(f"  Samples/sec: {trainer_stats.metrics.get('train_samples_per_second', 0):.2f}")

    # 6. Save with Unsloth merged save — NEVER naive save_pretrained
    print(f"Saving merged checkpoint to {output_dir}")
    model.save_pretrained_merged(
        output_dir,
        tokenizer,
        save_method="merged_16bit",
    )

    # 7. Post-save verification
    verified = _verify_checkpoint(output_dir, tokenizer, seed)
    if not verified:
        print("WARNING: Checkpoint verification failed — output may not produce valid briefs")
        logger.warning("SFT checkpoint verification failed for %s", output_dir)

    # 8. Return checkpoint path
    return output_dir


def _verify_checkpoint(
    checkpoint_dir: str,
    tokenizer,
    seed: int,
) -> bool:
    """Quick sanity check: load checkpoint, run one inference, check 6 section headers."""
    from unsloth import FastLanguageModel

    print("Verifying checkpoint...")

    try:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

        system_prompt = OperatingBriefPromptBuilder.SYSTEM_PROMPT
        full_prompt = (
            f"{_USER_TAG}\n{system_prompt}\n\n{_VERIFICATION_PROMPT}{_END_TAG}\n"
            f"{_ASSISTANT_TAG}\n"
        )

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=1.0,  # Required when do_sample=False but set for clarity
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Check all 6 required section headers are present
        missing = [s for s in _REQUIRED_SECTIONS if s not in generated]
        if missing:
            print(f"CHECKPOINT VERIFICATION FAILED — missing sections: {missing}")
            logger.warning("Verification missing sections: %s", missing)
            return False

        print("CHECKPOINT VERIFIED — all 6 sections present")
        return True

    except Exception as e:
        print(f"CHECKPOINT VERIFICATION FAILED — error: {e}")
        logger.warning("Verification error: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run SFT warm-start training")
    parser.add_argument("--model-id", default="google/gemma-4-26b-it")
    parser.add_argument("--output-dir", default="checkpoints/sft_v1")
    parser.add_argument("--data-dir", default="training/sft_data")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = run_sft(
        model_id=args.model_id,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"SFT complete. Checkpoint: {checkpoint}")
