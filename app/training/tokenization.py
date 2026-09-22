from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from app.data.dataset import TrainingExample
from app.environment.state import State
from app.training.prompting import build_completion, build_system_prompt, build_user_prompt


def tokenized_dataset(examples: list[TrainingExample], tokenizer: PreTrainedTokenizerBase) -> Dataset:
    rows = []
    for example in examples:
        state = State(task=example.task, available_tools=example.available_tools, history=example.history)
        system_prompt = build_system_prompt(example.available_tools)
        user_prompt = build_user_prompt(example.task, state)

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
