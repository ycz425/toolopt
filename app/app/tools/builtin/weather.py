from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class WeatherResult(BaseModel):
    city: str = Field(description="The city the weather is for.")
    temperature_f: float = Field(description="Temperature in Fahrenheit.")
    conditions: str = Field(description="Brief description of conditions, e.g. 'sunny', 'rainy'.")
    humidity_pct: float = Field(description="Relative humidity, as a percentage.")
    wind_mph: float = Field(description="Wind speed in miles per hour.")


_WEATHER_DATA: dict[str, WeatherResult] = {
    "toronto": WeatherResult(city="Toronto", temperature_f=52.0, conditions="cloudy", humidity_pct=70.0, wind_mph=8.0),
    "vancouver": WeatherResult(city="Vancouver", temperature_f=48.0, conditions="rainy", humidity_pct=85.0, wind_mph=6.0),
    "new york": WeatherResult(city="New York", temperature_f=61.0, conditions="partly cloudy", humidity_pct=55.0, wind_mph=12.0),
    "london": WeatherResult(city="London", temperature_f=57.0, conditions="overcast", humidity_pct=78.0, wind_mph=10.0),
    "tokyo": WeatherResult(city="Tokyo", temperature_f=68.0, conditions="sunny", humidity_pct=60.0, wind_mph=5.0),
    "paris": WeatherResult(city="Paris", temperature_f=59.0, conditions="clear", humidity_pct=65.0, wind_mph=9.0),
    "sydney": WeatherResult(city="Sydney", temperature_f=75.0, conditions="sunny", humidity_pct=50.0, wind_mph=11.0),
    "berlin": WeatherResult(city="Berlin", temperature_f=54.0, conditions="windy", humidity_pct=72.0, wind_mph=15.0),
    "chicago": WeatherResult(city="Chicago", temperature_f=49.0, conditions="windy", humidity_pct=68.0, wind_mph=18.0),
    "mumbai": WeatherResult(city="Mumbai", temperature_f=86.0, conditions="humid", humidity_pct=88.0, wind_mph=7.0),
    "singapore": WeatherResult(city="Singapore", temperature_f=88.0, conditions="humid", humidity_pct=82.0, wind_mph=6.0),
    "dubai": WeatherResult(city="Dubai", temperature_f=95.0, conditions="sunny", humidity_pct=35.0, wind_mph=9.0),
    "moscow": WeatherResult(city="Moscow", temperature_f=34.0, conditions="snowy", humidity_pct=80.0, wind_mph=13.0),
    "cairo": WeatherResult(city="Cairo", temperature_f=82.0, conditions="clear", humidity_pct=40.0, wind_mph=8.0),
    "mexico city": WeatherResult(city="Mexico City", temperature_f=66.0, conditions="partly cloudy", humidity_pct=58.0, wind_mph=7.0),
    "rio de janeiro": WeatherResult(city="Rio de Janeiro", temperature_f=79.0, conditions="sunny", humidity_pct=70.0, wind_mph=10.0),
    "seoul": WeatherResult(city="Seoul", temperature_f=58.0, conditions="cloudy", humidity_pct=62.0, wind_mph=8.0),
    "bangkok": WeatherResult(city="Bangkok", temperature_f=90.0, conditions="humid", humidity_pct=78.0, wind_mph=5.0),
    "rome": WeatherResult(city="Rome", temperature_f=64.0, conditions="sunny", humidity_pct=55.0, wind_mph=8.0),
    "amsterdam": WeatherResult(city="Amsterdam", temperature_f=53.0, conditions="rainy", humidity_pct=80.0, wind_mph=14.0),
}


class WeatherTool(Tool):
    metadata = ToolMetadata(
        name="weather",
        description="Look up current weather for a city.",
        parameters=[
            ToolParameter(
                name="city",
                type="string",
                description="City name, e.g. 'Toronto'.",
            )
        ],
        returns=WeatherResult.__name__,
    )

    def execute(self, city: str) -> WeatherResult:
        result = _WEATHER_DATA.get(city.strip().lower())
        if result is None:
            raise ValueError(f"no weather data available for '{city}'")
        return result

    def cost(self, city: str) -> float:
        return 0.02
