from pydantic import BaseModel, Field
from app.agents.base import GeminiAgent
from app.environment.task import Task


class JudgeResult(BaseModel):
    correct: bool = Field(description="Whether the agent's answer conveys the same information as the reference answer.")
    reasoning: str = Field(description="Brief explanation for the verdict, naming any value or conclusion that differs.")

class Judge(GeminiAgent):
    async def judge(self, task: Task, expected_answer: str, actual_answer: str) -> JudgeResult:
        prompt = (
            "You are comparing an AI agent's final answer to a reference answer for the same task.\n\n"
            f"Task (context only): {task.description}\n"
            f"Reference answer: {expected_answer}\n"
            f"Agent's answer: {actual_answer}\n\n"
            "The reference answer is ground truth. It was produced from the same tools and data the agent used, "
            "which can differ from the real world (simulated prices, weather, distances, exchange rates, and unit "
            "conventions). Do not check either answer against outside knowledge, and do not judge whether the answer "
            "fully completes the task: if the reference answer is brief, an equally brief agent answer is correct.\n\n"
            "Mark the agent's answer correct if it conveys the same information as the reference answer: the same "
            "values, names, dates, and conclusions. Ignore differences in phrasing, word order, capitalization, "
            "punctuation, and number formatting (e.g. \"680\" vs \"680.0\", \"$221.26\" vs \"221.26 dollars\"), and "
            "rounding that agrees at the reference answer's precision. Extra detail beyond the reference answer is "
            "fine as long as it doesn't contradict it. Mark it incorrect if any value or conclusion in the reference "
            "answer is missing, different, or contradicted."
        )
        return await self.generate_structured(prompt=prompt, schema=JudgeResult, label="judge")
