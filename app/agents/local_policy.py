import torch
from pydantic import ValidationError
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from app.agents.base import Policy
from app.environment.state import Action, State
from app.environment.task import Task
from app.tools.base import ToolMetadata
from app.training.prompting import build_system_prompt, build_user_prompt


class LocalPolicy(Policy):
    label = "local_policy"

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, max_new_tokens: int = 256, max_retries: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries

    async def select_action(
        self, task: Task, tools: list[ToolMetadata], state: State, temperature: float = 0
    ) -> Action:
        messages = [
            {"role": "system", "content": build_system_prompt(tools)},
            {"role": "user", "content": build_user_prompt(task, state)},
        ]

        validation_error: str | None = None
        for attempt in range(self.max_retries + 1):
            turn = list(messages)
            if validation_error:
                turn.append({
                    "role": "user",
                    "content": (
                        "Your previous output failed schema validation:\n\n"
                        f"{validation_error}\n\n"
                        "Return a corrected response that strictly matches the required schema."
                    ),
                })

            output_text = self._generate(turn, temperature)
            try:
                return Action.model_validate_json(output_text)
            except ValidationError as e:
                if attempt == self.max_retries:
                    raise
                validation_error = str(e)

    def _generate(self, messages: list[dict], temperature: float) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **generate_kwargs)

        new_tokens = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
