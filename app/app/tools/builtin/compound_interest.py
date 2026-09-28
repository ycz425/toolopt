from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class CompoundInterestResult(BaseModel):
    principal: float = Field(description="The initial amount.")
    rate_pct: float = Field(description="Annual interest rate, as a percentage.")
    years: int = Field(description="Number of years compounded.")
    final_amount: float = Field(description="The amount after compounding.")


class CompoundInterestTool(Tool):
    metadata = ToolMetadata(
        name="compound_interest",
        description="Compute the final amount after annual compound interest is applied.",
        parameters=[
            ToolParameter(name="principal", type="number", description="The initial amount."),
            ToolParameter(name="rate_pct", type="number", description="Annual interest rate as a percentage, e.g. 5 for 5%."),
            ToolParameter(name="years", type="integer", description="Number of years compounded."),
        ],
        returns=CompoundInterestResult.__name__,
    )

    def execute(self, principal: float, rate_pct: float, years: int) -> CompoundInterestResult:
        final_amount = principal * (1 + rate_pct / 100) ** years
        return CompoundInterestResult(
            principal=principal, rate_pct=rate_pct, years=years, final_amount=round(final_amount, 2)
        )

    def cost(self, principal: float, rate_pct: float, years: int) -> float:
        return 0.01
