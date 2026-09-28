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
            {'role': 'system', 'content': build_system_prompt(tools)},
            {'role': 'user', 'content': build_user_prompt(task, state)}
        ]

        for attempt in range(self.max_retries + 1):
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            ).to(self.model.device)

            output_ids = self.model.generate(
                inputs=prompt_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=(temperature != 0),
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )

            completion_ids = output_ids[0, len(prompt_ids):]

            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            try:
                return Action.model_validate_json(completion_text)
            except ValidationError as e:
                if attempt == self.max_retries:
                    raise
                messages.extend(
                    [
                        {'role': 'assistant', 'content': completion_text},
                        {'role': 'user', 'content':
                            (
                                "\n\nYour previous output failed schema validation:\n\n"
                                f"{str(e)}\n\n"
                                "Return a corrected response that strictly matches the required schema.\n"
                            )
                        }
                    ]
                )