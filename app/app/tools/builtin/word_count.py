from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class WordCountResult(BaseModel):
    words: int = Field(description="Number of whitespace-separated words in the text.")
    characters: int = Field(description="Number of characters in the text, including spaces.")


class WordCountTool(Tool):
    metadata = ToolMetadata(
        name="word_count",
        description="Count the words and characters in a piece of text.",
        parameters=[
            ToolParameter(name="text", type="string", description="The text to analyze."),
        ],
        returns=WordCountResult.__name__,
    )

    def execute(self, text: str) -> WordCountResult:
        return WordCountResult(words=len(text.split()), characters=len(text))

    def cost(self, text: str) -> float:
        return 0.01
