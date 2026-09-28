from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class TemperatureResult(BaseModel):
    value: float = Field(description="The original temperature value.")
    from_unit: str = Field(description="The unit converted from.")
    to_unit: str = Field(description="The unit converted to.")
    converted_value: float = Field(description="The temperature expressed in to_unit.")


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    from_unit, to_unit = from_unit.strip().lower(), to_unit.strip().lower()
    if from_unit == to_unit:
        return value

    if from_unit == "fahrenheit":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "celsius":
        celsius = value
    elif from_unit == "kelvin":
        celsius = value - 273.15
    else:
        raise ValueError(f"unknown temperature unit '{from_unit}'")

    if to_unit == "fahrenheit":
        return celsius * 9 / 5 + 32
    if to_unit == "celsius":
        return celsius
    if to_unit == "kelvin":
        return celsius + 273.15
    raise ValueError(f"unknown temperature unit '{to_unit}'")


class TemperatureConverterTool(Tool):
    metadata = ToolMetadata(
        name="temperature_converter",
        description="Convert a temperature between Celsius, Fahrenheit, and Kelvin.",
        parameters=[
            ToolParameter(name="value", type="number", description="The temperature value to convert."),
            ToolParameter(name="from_unit", type="string", description="One of 'celsius', 'fahrenheit', 'kelvin'."),
            ToolParameter(name="to_unit", type="string", description="One of 'celsius', 'fahrenheit', 'kelvin'."),
        ],
        returns=TemperatureResult.__name__,
    )

    def execute(self, value: float, from_unit: str, to_unit: str) -> TemperatureResult:
        return TemperatureResult(
            value=value,
            from_unit=from_unit,
            to_unit=to_unit,
            converted_value=round(convert_temperature(value, from_unit, to_unit), 2),
        )

    def cost(self, value: float, from_unit: str, to_unit: str) -> float:
        return 0.01
