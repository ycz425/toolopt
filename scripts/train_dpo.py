"""DPO on top of an SFT run: prefer the cheaper valid tool call (cost pairs) and avoid cheap tools outside their
coverage (correctness pairs), using the scripted preference pairs of the SFT run's dataset version.

    python -m scripts.train_dpo --sft-run experiments/runs/<sft run>

The SFT run's adapter is both the starting policy and, as a frozen copy TRL makes, the reference model. The output is
a normal run folder, so `scripts.benchmark --run <dpo run>` works unchanged.
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from trl import DPOConfig, DPOTrainer

from app.data.preference import load_preference_pairs
from app.training.config import DPOTrainingConfig, LoRATrainingConfig
from app.training.dpo import dpo_dataset, ensure_logprob_kernel
from app.training.lora import load_tokenizer, load_trained_adapter
from app.training.runs import RUNS_ROOT, read_run_info, write_run_info


def get_dpo_args(config: DPOTrainingConfig, output_dir: Path, warmup_steps: int) -> DPOConfig:
    return DPOConfig(
        output_dir=str(output_dir),
        seed=config.seed,

        beta=config.beta,
        loss_type=["sigmoid", "sft"] if config.sft_loss_weight > 0 else ["sigmoid"],
        loss_weights=[1.0, config.sft_loss_weight] if config.sft_loss_weight > 0 else None,
        max_length=config.max_length,

        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,

        bf16=config.bf16,

        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,

        logging_steps=config.logging_steps,
        report_to=config.report_to,
    )


def main():
    parser = argparse.ArgumentParser(description="Train DPO on top of an SFT run, using its dataset's preference pairs.")
    parser.add_argument("--sft-run", type=Path, required=True, help="SFT run directory (experiments/runs/<run>) to start from")
    args = parser.parse_args()

    # The SFT run fixes the model (base model, attention, LoRA shape) and the dataset version: DPO must use the same
    # ones, so they're read from its run_info rather than from defaults.
    sft_run_dir: Path = args.sft_run
    sft_run_info = read_run_info(sft_run_dir)
    sft_config = LoRATrainingConfig.model_validate(sft_run_info["config"])
    dataset_dir = Path(sft_run_info["dataset_dir"])

    config = DPOTrainingConfig()
    run_dir = RUNS_ROOT / f"{datetime.now():%Y-%m-%d_%H%M%S}_{dataset_dir.name}_dpo"
    print(f"DPO on {dataset_dir} from {sft_run_dir}; writing run to {run_dir}")

    model = load_trained_adapter(sft_config, str(sft_run_dir / "adapter"), trainable=True)
    tokenizer = load_tokenizer(sft_config)

    train_pairs = load_preference_pairs(dataset_dir / "dpo_pairs.jsonl")
    val_pairs = load_preference_pairs(dataset_dir / "dpo_val_pairs.jsonl")
    train_dataset = dpo_dataset(train_pairs)
    eval_dataset = dpo_dataset(val_pairs)

    manifest_path = dataset_dir / "manifest.json"
    run_info = {
        "run": run_dir.name,
        "stage": "dpo",
        "dataset_version": sft_run_info["dataset_version"],
        "dataset_dir": str(dataset_dir),
        "sft_run": str(sft_run_dir),
        "sft_run_info": sft_run_info,
        "dataset_manifest": json.loads(manifest_path.read_text()) if manifest_path.exists() else None,
        "started": datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "counts": {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "train_pairs_by_kind": {k: sum(p.kind == k for p in train_pairs) for k in ("cost", "correctness")},
        },
        # base_model and attn_implementation come from the SFT run; the trainer settings are DPO's own.
        "config": {"base_model": sft_config.base_model, "attn_implementation": sft_config.attn_implementation, **config.model_dump()},
    }
    write_run_info(run_dir, run_info)

    # TrainingArguments (which DPOConfig extends) no longer accepts warmup_ratio, so convert it to a step count.
    steps_per_epoch = math.ceil(len(train_dataset) / (config.batch_size * config.gradient_accumulation_steps))
    warmup_steps = round(config.warmup_ratio * steps_per_epoch * config.num_epochs)

    # No ref_model and no peft_config: given a model that already has a trained adapter, TRL copies it into a frozen
    # "ref" adapter and uses that as the reference, so the reference is exactly the SFT model.
    ensure_logprob_kernel()
    trainer = DPOTrainer(
        model=model,
        args=get_dpo_args(config, run_dir, warmup_steps),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    dropped = len(train_pairs) - len(trainer.train_dataset)
    if dropped:
        print(f"WARNING: TRL dropped {dropped} training pairs (likely longer than max_length={config.max_length})")
    run_info["counts"]["train_pairs_after_tokenization"] = len(trainer.train_dataset)
    write_run_info(run_dir, run_info)

    trainer.train()

    # With load_best_model_at_end, the policy is now the best checkpoint. Save only the trained ("default") adapter, at
    # the top of adapter/, which is where benchmark.py looks; the frozen "ref" copy isn't needed.
    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir), selected_adapters=["default"])
    tokenizer.save_pretrained(str(adapter_dir))

    run_info["finished"] = datetime.now().isoformat(timespec="seconds")
    run_info["best_checkpoint"] = trainer.state.best_model_checkpoint
    run_info["best_metric"] = trainer.state.best_metric
    write_run_info(run_dir, run_info)
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
