from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class GroupResult(BaseModel):
    group: str = Field(description="The group key value.")
    total: float = Field(description="Sum of the value field for this group.")
    count: int = Field(description="Number of records in this group.")


class AggregateRecordsResult(BaseModel):
    groups: list[GroupResult] = Field(description="One entry per distinct group value, sorted by group name.")


class AggregateRecordsTool(Tool):
    metadata = ToolMetadata(
        name="aggregate_records",
        description=(
            "Group a list of records by category and sum a numeric value within each group. "
            "Each record is an object with 'category' (string) and 'value' (number) fields."
        ),
        parameters=[
            ToolParameter(
                name="records",
                type="array",
                description="List of records, each an object with 'category' and 'value' fields.",
            ),
        ],
        returns=AggregateRecordsResult.__name__,
    )

    def execute(self, records: list[dict]) -> AggregateRecordsResult:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for r in records:
            key = r["category"]
            totals[key] = totals.get(key, 0.0) + r["value"]
            counts[key] = counts.get(key, 0) + 1
        groups = [GroupResult(group=k, total=round(totals[k], 2), count=counts[k]) for k in sorted(totals)]
        return AggregateRecordsResult(groups=groups)

    def cost(self, records: list[dict]) -> float:
        return 0.02
