from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter, ToolPricing


class PrimeResult(BaseModel):
    is_prime: bool = Field(description="Whether the number is prime.")


class PrimeCheckerTool(Tool):
    metadata = ToolMetadata(
        name="prime_checker",
        description="Check whether an integer is a prime number.",
        pricing=ToolPricing(base=0.01),
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


SMALL_PRIME_LIMIT = 9_999


class SmallPrimeCheckerTool(Tool):
    metadata = ToolMetadata(
        name="small_prime_checker",
        description=f"Check whether an integer is a prime number. Only supports integers up to {SMALL_PRIME_LIMIT:,}.",
        pricing=ToolPricing(base=0.004),
        parameters=[
            ToolParameter(name="number", type="integer", description=f"The integer to check, at most {SMALL_PRIME_LIMIT:,}."),
        ],
        returns=PrimeResult.__name__,
    )

    def execute(self, number: int) -> PrimeResult:
        if int(number) > SMALL_PRIME_LIMIT:
            raise ValueError(f"small_prime_checker only supports integers up to {SMALL_PRIME_LIMIT:,}; got {number}")
        return PrimeCheckerTool().execute(number)
