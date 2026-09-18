from pydantic import BaseModel, Field


class LoRATrainingConfig(BaseModel):
    base_model: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="HuggingFace model identifier for the base model to fine-tune.",
    )

    lora_r: int = Field(
        default=16, description="Rank of the LoRA update matrices. Higher = more capacity and parameters."
    )
    lora_alpha: int = Field(
        default=32, description="LoRA scaling factor; the effective update is scaled by lora_alpha / lora_r."
    )
    lora_dropout: float = Field(
        default=0.05, description="Dropout applied on the LoRA path during training, for regularization."
    )
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Names of the weight matrices LoRA adapters attach to (Qwen2/Qwen2.5 attention projections).",
    )

    use_4bit: bool = Field(
        default=False,
        description="Load the base model in 4-bit (QLoRA) instead of bf16. Enable for constrained local hardware.",
    )

    learning_rate: float = Field(default=2e-4, description="Learning rate for the LoRA parameters.")
    num_epochs: int = Field(default=3, description="Number of passes over the training data.")
    batch_size: int = Field(default=8, description="Per-device training batch size.")
    gradient_accumulation_steps: int = Field(
        default=1, description="Number of steps to accumulate gradients over before an optimizer step."
    )
    max_seq_length: int = Field(
        default=2048, description="Maximum tokenized sequence length; longer prompts are truncated."
    )
    warmup_ratio: float = Field(default=0.03, description="Fraction of total steps used for learning-rate warmup.")
    optim: str = Field(default="adamw_torch_fused", description="Optimizer used for training.")
    weight_decay: float = Field(default=0.01, description="Weight decay applied by the optimizer.")

    output_dir: str = Field(
        default="experiments/lora/adapter", description="Directory to save the trained LoRA adapter to."
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    bf16: bool = Field(
        default=True,
        description="Train in bf16 precision (H100). Typically paired with use_4bit=False.",
    )
    eval_strategy: str = Field(
        default="epoch", description="When to run evaluation: 'epoch', 'steps', or 'no'."
    )
    save_strategy: str = Field(
        default="epoch",
        description="When to save checkpoints. Must be compatible with eval_strategy for load_best_model_at_end to work.",
    )
    load_best_model_at_end: bool = Field(
        default=True,
        description="Keep the checkpoint with the best eval metric at the end of training, not just the last epoch.",
    )
    metric_for_best_model: str = Field(
        default="eval_loss", description="Metric used to decide which checkpoint is 'best'."
    )
    remove_unused_columns: bool = Field(
        default=False,
        description="Must be False for PEFT models -- Trainer's auto column-pruning can incorrectly strip 'labels'.",
    )
    logging_steps: int = Field(default=10, description="How often (in steps) to log training loss.")
    report_to: str = Field(
        default="none", description="Experiment tracking integration ('none', 'wandb', 'tensorboard', ...)."
    )
