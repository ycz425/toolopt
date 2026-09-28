from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class PrimeResult(BaseModel):
    is_prime: bool = Field(description="Whether the number is prime.")


class PrimeCheckerTool(Tool):
    metadata = ToolMetadata(
        name="prime_checker",
        description="Check whether an integer is a prime number.",
        parameters=[
            ToolParameter(name="number", type="integer", description="The integer to check."),
        ],
        returns=PrimeResult.__name__,
    )

    def execute(self, number: int) -> PrimeResult:
        number = int(number)
        if number < 2:
            return PrimeResult(is_prime=False)
        for divisor in range(2, int(number ** 0.5) + 1):
            if number % divisor == 0:
                return PrimeResult(is_prime=False)
        return PrimeResult(is_prime=True)

    def cost(self, number: int) -> float:
        return 0.01
