from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter
from app.tools.builtin.weather import _WEATHER_DATA

_CONDITIONS_CYCLE = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]


def _deterministic_offset(s: str) -> int:
    return sum(ord(c) for c in s)


class ForecastDay(BaseModel):
    day: int = Field(description="Day offset from today (1 = tomorrow).")
    temperature_f: float = Field(description="Forecasted temperature in Fahrenheit.")
    conditions: str = Field(description="Forecasted conditions.")


class WeatherForecastResult(BaseModel):
    city: str = Field(description="The city forecasted.")
    forecast: list[ForecastDay] = Field(description="Day-by-day forecast.")


class WeatherForecastTool(Tool):
    metadata = ToolMetadata(
        name="weather_forecast",
        description="Get a multi-day weather forecast for a city -- more detailed (and costlier) than the basic weather tool, which only gives current conditions.",
        parameters=[
            ToolParameter(name="city", type="string", description="City name, e.g. 'Toronto'."),
            ToolParameter(name="days", type="integer", description="Number of days to forecast (1-5)."),
        ],
        returns=WeatherForecastResult.__name__,
    )

    def execute(self, city: str, days: int) -> WeatherForecastResult:
        base = _WEATHER_DATA.get(city.strip().lower())
        if base is None:
            raise ValueError(f"no weather data available for '{city}'")
        if not (1 <= days <= 5):
            raise ValueError("days must be between 1 and 5")
        offset = _deterministic_offset(city.strip().lower())
        forecast = [
            ForecastDay(
                day=d,
                temperature_f=round(base.temperature_f + (d * 1.5 - 3), 1),
                conditions=_CONDITIONS_CYCLE[(offset + d) % len(_CONDITIONS_CYCLE)],
            )
            for d in range(1, days + 1)
        ]
        return WeatherForecastResult(city=base.city, forecast=forecast)

    def cost(self, city: str, days: int) -> float:
        return 0.06
