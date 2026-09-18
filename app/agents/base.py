import dotenv
from google import genai
from typing import TypeVar
import os
from pydantic import BaseModel, ValidationError
from datetime import datetime

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

T = TypeVar("T", bound=BaseModel)


class LLMAgent:
    def __init__(self, model: str = "gemini-3.1-flash-lite", verbose: bool = False):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        self.verbose = verbose

    async def generate_structured(self, prompt: str, schema: type[T], max_retries: int = 5, label: str = "", temperature: float = 0) -> T:
        validation_error: str | None = None
        for attempt in range(max_retries + 1):
            full_prompt = prompt
            if validation_error:
                full_prompt += (
                    "\n\nYour previous output failed schema validation:\n\n"
                    f"{validation_error}\n\n"
                    "Return a corrected response that strictly matches the required schema.\n"
                )

            interaction = await self.client.aio.interactions.create(
                model=self.model,
                input=full_prompt,
                generation_config={
                    'thinking_level': 'low',
                    'temperature': temperature,
                },
                response_format={
                    'mime_type': 'application/json',
                    'schema': schema.model_json_schema(),
                }
            )

            try:
                return schema.model_validate_json(interaction.output_text)
            except ValidationError as e:
                if attempt == max_retries:
                    raise
                validation_error = str(e)
                if self.verbose:
                    print(f'{datetime.now()}     {label} model validation failed - retrying... (attempt: {attempt + 1}/{max_retries})')
