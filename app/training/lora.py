import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase

from app.training.config import LoRATrainingConfig


def apply_lora(model, config: LoRATrainingConfig):
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model(config: LoRATrainingConfig):
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model, quantization_config=quantization_config, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        )
    return apply_lora(model, config)


def load_tokenizer(config: LoRATrainingConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.padding_side = "right"
    return tokenizer

