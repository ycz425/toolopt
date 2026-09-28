from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class FinishResult(BaseModel):
    answer: str = Field(description="The agent's final answer to the task.")


class FinishTool(Tool):
    metadata = ToolMetadata(
        name="finish",
        description="Call this when you have completed the task, with your final answer.",
        parameters=[
            ToolParameter(
                name="answer",
                type="string",
                description="The agent's final answer to the task.",
            )
        ],
        returns=FinishResult.__name__,
    )

    def execute(self, answer: str) -> FinishResult:
        return FinishResult(answer=answer)

    def cost(self, answer: str) -> float:
        return 0.0
