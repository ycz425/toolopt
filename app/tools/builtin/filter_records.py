from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter, ToolPricing


class FilterRecordsResult(BaseModel):
    matching: list[dict] = Field(description="Records whose score met or exceeded the threshold.")


class FilterRecordsTool(Tool):
    metadata = ToolMetadata(
        name="filter_records",
        description=(
            "Filter a list of records down to only those meeting a minimum score. Each record is an "
            "object with a 'name' (string) and 'score' (number) field."
        ),
        pricing=ToolPricing(base=0.01),
        parameters=[
            ToolParameter(
                name="records",
                type="array",
                description="List of records, each an object with 'name' and 'score' fields.",
            ),
            ToolParameter(name="threshold", type="number", description="Minimum score to include."),
        ],
        returns=FilterRecordsResult.__name__,
    )

    def execute(self, records: list[dict], threshold: float) -> FilterRecordsResult:
        matching = [r for r in records if r.get("score", float("-inf")) >= threshold]
        return FilterRecordsResult(matching=matching)
