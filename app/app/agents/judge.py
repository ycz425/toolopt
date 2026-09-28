from pydantic import BaseModel, Field
from app.agents.base import GeminiAgent
from app.environment.task import Task


class JudgeResult(BaseModel):
    correct: bool = Field(description="Whether the answer correctly addresses the task.")
    reasoning: str = Field(description="Brief explanation for the verdict.")

class Judge(GeminiAgent):
    async def judge(self, task: Task, expected_answer: str, actual_answer: str) -> JudgeResult:
        prompt = (
            f"Task: {task.description}\n"
            f"Expected answer: {expected_answer}\n"
            f"Agent's answer: {actual_answer}\n\n"
            "Does the agent's answer correctly address the task, "
            "allowing for different phrasing of the same fact?"
        )
        return await self.generate_structured(prompt=prompt, schema=JudgeResult, label="judge")