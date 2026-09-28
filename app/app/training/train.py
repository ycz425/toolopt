from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from app.data.dataset import build_dataset
from app.training.config import LoRATrainingConfig
from app.training.lora import load_model, load_tokenizer
from app.training.tokenization import tokenized_dataset


def get_training_args(config: LoRATrainingConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.output_dir,
        seed=config.seed,

        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
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


def main():
    config = LoRATrainingConfig()

    model = load_model(config=config)
    tokenizer = load_tokenizer(config=config)

    examples = build_dataset("/Users/ycz425/Desktop/projects/toolopt/datasets/trajectories/trajectories.jsonl")
    dataset = tokenized_dataset(examples, tokenizer)

    split_dataset = dataset.train_test_split(test_size=0.1, seed=config.seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    trainer = Trainer(
        model,
        args=get_training_args(config),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model, padding=True, label_pad_token_id=-100),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
