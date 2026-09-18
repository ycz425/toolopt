from datetime import date, timedelta

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class AddDaysResult(BaseModel):
    result_date: str = Field(description="The resulting date, in YYYY-MM-DD format.")


class AddDaysTool(Tool):
    metadata = ToolMetadata(
        name="add_days",
        description="Add (or subtract, with a negative number) a number of days to a date.",
        parameters=[
            ToolParameter(name="start_date", type="string", description="Starting date, in YYYY-MM-DD format."),
            ToolParameter(name="days", type="integer", description="Number of days to add (negative to subtract)."),
        ],
        returns=AddDaysResult.__name__,
    )

    def execute(self, start_date: str, days: int) -> AddDaysResult:
        result = date.fromisoformat(start_date) + timedelta(days=days)
        return AddDaysResult(result_date=result.isoformat())

    def cost(self, start_date: str, days: int) -> float:
        return 0.01
