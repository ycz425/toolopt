from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter

_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


class RomanResult(BaseModel):
    roman: str = Field(description="The number expressed as a Roman numeral.")


class RomanNumeralTool(Tool):
    metadata = ToolMetadata(
        name="roman_numeral",
        description="Convert an integer between 1 and 3999 to a Roman numeral.",
        parameters=[
            ToolParameter(name="value", type="integer", description="Integer between 1 and 3999."),
        ],
        returns=RomanResult.__name__,
    )

    def execute(self, value: int) -> RomanResult:
        value = int(value)
        if not (1 <= value <= 3999):
            raise ValueError("value must be between 1 and 3999")
        result = ""
        for amount, symbol in _VALUES:
            while value >= amount:
                result += symbol
                value -= amount
        return RomanResult(roman=result)

    def cost(self, value: int) -> float:
        return 0.01
