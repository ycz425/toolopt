import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from app.data.dataset import load_trajectories, trajectory_to_examples
from app.data.versions import resolve_version_dir
from app.training.config import LoRATrainingConfig
from app.training.lora import load_model, load_tokenizer
from app.training.tokenization import tokenized_dataset

RUNS_ROOT = Path("experiments/runs")


def get_training_args(config: LoRATrainingConfig, warmup_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.output_dir,
        seed=config.seed,

        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,

        bf16=config.bf16,

        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,

        remove_unused_columns=config.remove_unused_columns,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
    )


def write_run_info(run_dir: Path, info: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a LoRA adapter on a dataset version's trajectories.")
    parser.add_argument("--dataset", help="Dataset version to train on, e.g. v2 (default: the latest)")
    args = parser.parse_args()

    dataset_dir = resolve_version_dir(args.dataset)
    run_dir = RUNS_ROOT / f"{datetime.now():%Y-%m-%d_%H%M%S}_{dataset_dir.name}"
    config = LoRATrainingConfig(output_dir=str(run_dir))
    print(f"Training on {dataset_dir}; writing run to {run_dir}")

    model = load_model(config=config)
    tokenizer = load_tokenizer(config=config)

    train_trajectories = [t for t in load_trajectories(dataset_dir / "train_trajectories.jsonl") if t.success]
    val_trajectories = [t for t in load_trajectories(dataset_dir / "val_trajectories.jsonl") if t.success]
    train_dataset = tokenized_dataset([e for t in train_trajectories for e in trajectory_to_examples(t)], tokenizer)
    eval_dataset = tokenized_dataset([e for t in val_trajectories for e in trajectory_to_examples(t)], tokenizer)

    manifest_path = dataset_dir / "manifest.json"
    run_info = {
        "run": run_dir.name,
        "dataset_version": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "dataset_manifest": json.loads(manifest_path.read_text()) if manifest_path.exists() else None,
        "started": datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "counts": {
            "train_trajectories": len(train_trajectories),
            "val_trajectories": len(val_trajectories),
            "train_examples": len(train_dataset),
            "val_examples": len(eval_dataset),
        },
        "config": config.model_dump(),
    }
    write_run_info(run_dir, run_info)

    # TrainingArguments no longer accepts warmup_ratio, so convert it to a step count.
    steps_per_epoch = math.ceil(len(train_dataset) / (config.batch_size * config.gradient_accumulation_steps))
    warmup_steps = round(config.warmup_ratio * steps_per_epoch * config.num_epochs)

    trainer = Trainer(
        model,
        args=get_training_args(config, warmup_steps),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model, padding=True, label_pad_token_id=-100),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # With load_best_model_at_end, the model is now the best checkpoint; save it where the benchmark looks.
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    run_info["finished"] = datetime.now().isoformat(timespec="seconds")
    run_info["best_checkpoint"] = trainer.state.best_model_checkpoint
    run_info["best_metric"] = trainer.state.best_metric
    write_run_info(run_dir, run_info)
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
