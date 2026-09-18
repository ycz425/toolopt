from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from app.data.dataset import TrainingExample
from app.training.prompting import build_completion, build_system_prompt, build_user_prompt


def tokenized_dataset(examples: list[TrainingExample], tokenizer: PreTrainedTokenizerBase) -> Dataset:
    rows = []
    for example in examples:
        system_prompt = build_system_prompt(example)
        user_prompt = build_user_prompt(example)

        full_ids = tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
                {'role': 'assistant', 'content': build_completion(example)}
            ],
            tokenize=True,
            add_generation_prompt=False
        )

        prompt_ids = tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            tokenize=True,
            add_generation_prompt=True
        )

        rows.append({
            'input_ids': full_ids,
            'labels': [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
            'attention_mask': [1] * len(full_ids),
        })

    return Dataset.from_list(rows)
