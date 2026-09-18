from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class ConversionResult(BaseModel):
    value: float = Field(description="The original numeric value.")
    from_unit: str = Field(description="The unit converted from.")
    to_unit: str = Field(description="The unit converted to.")
    converted_value: float = Field(description="The value expressed in to_unit.")


_CONVERSION_RATES: dict[tuple[str, str], float] = {
    ("km", "miles"): 0.621371,
    ("kg", "lbs"): 2.20462,
    ("usd", "eur"): 0.92,
    ("usd", "gbp"): 0.79,
    ("usd", "jpy"): 149.5,
    ("usd", "cad"): 1.37,
    ("usd", "aud"): 1.52,
    ("usd", "chf"): 0.88,
    ("liters", "gallons"): 0.264172,
    ("meters", "feet"): 3.28084,
    ("cm", "inches"): 0.393701,
    ("miles", "meters"): 1609.34,
    ("acres", "hectares"): 0.404686,
    ("usd", "inr"): 83.5,
    ("usd", "cny"): 7.24,
    ("usd", "mxn"): 17.1,
    ("oz", "grams"): 28.3495,
    ("tons", "kg"): 907.185,
    ("yards", "meters"): 0.9144,
    ("stone", "kg"): 6.35029,
}


def _rate(from_unit: str, to_unit: str) -> float:
    from_unit, to_unit = from_unit.strip().lower(), to_unit.strip().lower()
    if from_unit == to_unit:
        return 1.0
    if (from_unit, to_unit) in _CONVERSION_RATES:
        return _CONVERSION_RATES[(from_unit, to_unit)]
    if (to_unit, from_unit) in _CONVERSION_RATES:
        return 1.0 / _CONVERSION_RATES[(to_unit, from_unit)]
    raise ValueError(f"no conversion rate available between '{from_unit}' and '{to_unit}'")


class ConverterTool(Tool):
    metadata = ToolMetadata(
        name="converter",
        description="Convert a numeric value between units or currencies (e.g. km to miles, USD to EUR).",
        parameters=[
            ToolParameter(name="value", type="number", description="The numeric value to convert."),
            ToolParameter(name="from_unit", type="string", description="The unit to convert from, e.g. 'km', 'usd'."),
            ToolParameter(name="to_unit", type="string", description="The unit to convert to, e.g. 'miles', 'eur'."),
        ],
        returns=ConversionResult.__name__,
    )

    def execute(self, value: float, from_unit: str, to_unit: str) -> ConversionResult:
        return ConversionResult(
            value=value,
            from_unit=from_unit,
            to_unit=to_unit,
            converted_value=value * _rate(from_unit, to_unit),
        )

    def cost(self, value: float, from_unit: str, to_unit: str) -> float:
        return 0.01
