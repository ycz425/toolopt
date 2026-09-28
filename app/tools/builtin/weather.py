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
    "los angeles": WeatherResult(city="Los Angeles", temperature_f=76.0, conditions="sunny", humidity_pct=47.0, wind_mph=5.0),
    "san francisco": WeatherResult(city="San Francisco", temperature_f=57.0, conditions="partly cloudy", humidity_pct=61.0, wind_mph=5.0),
    "seattle": WeatherResult(city="Seattle", temperature_f=58.0, conditions="cloudy", humidity_pct=81.0, wind_mph=8.0),
    "boston": WeatherResult(city="Boston", temperature_f=47.0, conditions="rainy", humidity_pct=68.0, wind_mph=7.0),
    "miami": WeatherResult(city="Miami", temperature_f=83.0, conditions="humid", humidity_pct=87.0, wind_mph=7.0),
    "houston": WeatherResult(city="Houston", temperature_f=73.0, conditions="humid", humidity_pct=67.0, wind_mph=5.0),
    "dallas": WeatherResult(city="Dallas", temperature_f=78.0, conditions="sunny", humidity_pct=42.0, wind_mph=5.0),
    "denver": WeatherResult(city="Denver", temperature_f=57.0, conditions="sunny", humidity_pct=29.0, wind_mph=12.0),
    "phoenix": WeatherResult(city="Phoenix", temperature_f=94.0, conditions="sunny", humidity_pct=28.0, wind_mph=10.0),
    "atlanta": WeatherResult(city="Atlanta", temperature_f=77.0, conditions="humid", humidity_pct=78.0, wind_mph=8.0),
    "montreal": WeatherResult(city="Montreal", temperature_f=29.0, conditions="snowy", humidity_pct=77.0, wind_mph=19.0),
    "calgary": WeatherResult(city="Calgary", temperature_f=20.0, conditions="clear", humidity_pct=61.0, wind_mph=17.0),
    "ottawa": WeatherResult(city="Ottawa", temperature_f=24.0, conditions="windy", humidity_pct=81.0, wind_mph=16.0),
    "madrid": WeatherResult(city="Madrid", temperature_f=79.0, conditions="clear", humidity_pct=49.0, wind_mph=12.0),
    "barcelona": WeatherResult(city="Barcelona", temperature_f=66.0, conditions="cloudy", humidity_pct=57.0, wind_mph=7.0),
    "lisbon": WeatherResult(city="Lisbon", temperature_f=62.0, conditions="partly cloudy", humidity_pct=68.0, wind_mph=9.0),
    "dublin": WeatherResult(city="Dublin", temperature_f=52.0, conditions="overcast", humidity_pct=85.0, wind_mph=15.0),
    "edinburgh": WeatherResult(city="Edinburgh", temperature_f=48.0, conditions="cloudy", humidity_pct=77.0, wind_mph=9.0),
    "oslo": WeatherResult(city="Oslo", temperature_f=34.0, conditions="windy", humidity_pct=65.0, wind_mph=20.0),
    "stockholm": WeatherResult(city="Stockholm", temperature_f=28.0, conditions="cloudy", humidity_pct=75.0, wind_mph=14.0),
    "copenhagen": WeatherResult(city="Copenhagen", temperature_f=46.0, conditions="cloudy", humidity_pct=72.0, wind_mph=15.0),
    "helsinki": WeatherResult(city="Helsinki", temperature_f=28.0, conditions="overcast", humidity_pct=82.0, wind_mph=13.0),
    "vienna": WeatherResult(city="Vienna", temperature_f=64.0, conditions="rainy", humidity_pct=73.0, wind_mph=13.0),
    "prague": WeatherResult(city="Prague", temperature_f=47.0, conditions="cloudy", humidity_pct=63.0, wind_mph=13.0),
    "warsaw": WeatherResult(city="Warsaw", temperature_f=47.0, conditions="cloudy", humidity_pct=78.0, wind_mph=10.0),
    "budapest": WeatherResult(city="Budapest", temperature_f=63.0, conditions="rainy", humidity_pct=64.0, wind_mph=12.0),
    "athens": WeatherResult(city="Athens", temperature_f=77.0, conditions="sunny", humidity_pct=49.0, wind_mph=10.0),
    "istanbul": WeatherResult(city="Istanbul", temperature_f=60.0, conditions="partly cloudy", humidity_pct=65.0, wind_mph=5.0),
    "zurich": WeatherResult(city="Zurich", temperature_f=51.0, conditions="overcast", humidity_pct=59.0, wind_mph=9.0),
    "brussels": WeatherResult(city="Brussels", temperature_f=50.0, conditions="overcast", humidity_pct=90.0, wind_mph=9.0),
    "munich": WeatherResult(city="Munich", temperature_f=50.0, conditions="rainy", humidity_pct=67.0, wind_mph=14.0),
    "milan": WeatherResult(city="Milan", temperature_f=63.0, conditions="sunny", humidity_pct=63.0, wind_mph=13.0),
    "reykjavik": WeatherResult(city="Reykjavik", temperature_f=26.0, conditions="windy", humidity_pct=71.0, wind_mph=18.0),
    "kyiv": WeatherResult(city="Kyiv", temperature_f=57.0, conditions="partly cloudy", humidity_pct=59.0, wind_mph=7.0),
    "beijing": WeatherResult(city="Beijing", temperature_f=45.0, conditions="sunny", humidity_pct=27.0, wind_mph=16.0),
    "shanghai": WeatherResult(city="Shanghai", temperature_f=79.0, conditions="humid", humidity_pct=75.0, wind_mph=7.0),
    "hong kong": WeatherResult(city="Hong Kong", temperature_f=84.0, conditions="rainy", humidity_pct=70.0, wind_mph=5.0),
    "taipei": WeatherResult(city="Taipei", temperature_f=86.0, conditions="rainy", humidity_pct=88.0, wind_mph=6.0),
    "osaka": WeatherResult(city="Osaka", temperature_f=59.0, conditions="partly cloudy", humidity_pct=64.0, wind_mph=13.0),
    "manila": WeatherResult(city="Manila", temperature_f=86.0, conditions="partly cloudy", humidity_pct=82.0, wind_mph=7.0),
    "jakarta": WeatherResult(city="Jakarta", temperature_f=81.0, conditions="partly cloudy", humidity_pct=82.0, wind_mph=4.0),
    "kuala lumpur": WeatherResult(city="Kuala Lumpur", temperature_f=83.0, conditions="humid", humidity_pct=76.0, wind_mph=7.0),
    "hanoi": WeatherResult(city="Hanoi", temperature_f=77.0, conditions="humid", humidity_pct=70.0, wind_mph=5.0),
    "delhi": WeatherResult(city="Delhi", temperature_f=89.0, conditions="sunny", humidity_pct=48.0, wind_mph=6.0),
    "karachi": WeatherResult(city="Karachi", temperature_f=96.0, conditions="sunny", humidity_pct=60.0, wind_mph=7.0),
    "dhaka": WeatherResult(city="Dhaka", temperature_f=89.0, conditions="humid", humidity_pct=72.0, wind_mph=10.0),
    "kathmandu": WeatherResult(city="Kathmandu", temperature_f=61.0, conditions="clear", humidity_pct=54.0, wind_mph=9.0),
    "colombo": WeatherResult(city="Colombo", temperature_f=85.0, conditions="rainy", humidity_pct=85.0, wind_mph=4.0),
    "tehran": WeatherResult(city="Tehran", temperature_f=43.0, conditions="partly cloudy", humidity_pct=34.0, wind_mph=13.0),
    "riyadh": WeatherResult(city="Riyadh", temperature_f=105.0, conditions="clear", humidity_pct=12.0, wind_mph=8.0),
    "doha": WeatherResult(city="Doha", temperature_f=93.0, conditions="clear", humidity_pct=18.0, wind_mph=13.0),
    "tel aviv": WeatherResult(city="Tel Aviv", temperature_f=71.0, conditions="partly cloudy", humidity_pct=35.0, wind_mph=8.0),
    "nairobi": WeatherResult(city="Nairobi", temperature_f=71.0, conditions="cloudy", humidity_pct=54.0, wind_mph=13.0),
    "lagos": WeatherResult(city="Lagos", temperature_f=80.0, conditions="rainy", humidity_pct=72.0, wind_mph=9.0),
    "johannesburg": WeatherResult(city="Johannesburg", temperature_f=63.0, conditions="cloudy", humidity_pct=55.0, wind_mph=10.0),
    "cape town": WeatherResult(city="Cape Town", temperature_f=62.0, conditions="cloudy", humidity_pct=70.0, wind_mph=8.0),
    "casablanca": WeatherResult(city="Casablanca", temperature_f=61.0, conditions="sunny", humidity_pct=62.0, wind_mph=8.0),
    "addis ababa": WeatherResult(city="Addis Ababa", temperature_f=61.0, conditions="clear", humidity_pct=61.0, wind_mph=5.0),
    "accra": WeatherResult(city="Accra", temperature_f=80.0, conditions="rainy", humidity_pct=85.0, wind_mph=6.0),
    "lima": WeatherResult(city="Lima", temperature_f=61.0, conditions="cloudy", humidity_pct=64.0, wind_mph=10.0),
    "bogota": WeatherResult(city="Bogota", temperature_f=66.0, conditions="partly cloudy", humidity_pct=57.0, wind_mph=6.0),
    "santiago": WeatherResult(city="Santiago", temperature_f=73.0, conditions="clear", humidity_pct=41.0, wind_mph=10.0),
    "buenos aires": WeatherResult(city="Buenos Aires", temperature_f=61.0, conditions="clear", humidity_pct=69.0, wind_mph=5.0),
    "sao paulo": WeatherResult(city="Sao Paulo", temperature_f=81.0, conditions="partly cloudy", humidity_pct=46.0, wind_mph=6.0),
    "caracas": WeatherResult(city="Caracas", temperature_f=90.0, conditions="humid", humidity_pct=82.0, wind_mph=10.0),
    "havana": WeatherResult(city="Havana", temperature_f=91.0, conditions="sunny", humidity_pct=85.0, wind_mph=5.0),
    "auckland": WeatherResult(city="Auckland", temperature_f=50.0, conditions="cloudy", humidity_pct=85.0, wind_mph=9.0),
    "melbourne": WeatherResult(city="Melbourne", temperature_f=67.0, conditions="clear", humidity_pct=62.0, wind_mph=6.0),
    "perth": WeatherResult(city="Perth", temperature_f=71.0, conditions="sunny", humidity_pct=39.0, wind_mph=5.0),
    "brisbane": WeatherResult(city="Brisbane", temperature_f=70.0, conditions="partly cloudy", humidity_pct=49.0, wind_mph=7.0),
    "honolulu": WeatherResult(city="Honolulu", temperature_f=89.0, conditions="partly cloudy", humidity_pct=81.0, wind_mph=5.0),
    "anchorage": WeatherResult(city="Anchorage", temperature_f=35.0, conditions="clear", humidity_pct=64.0, wind_mph=8.0),
    "las vegas": WeatherResult(city="Las Vegas", temperature_f=90.0, conditions="sunny", humidity_pct=26.0, wind_mph=8.0),
    "new orleans": WeatherResult(city="New Orleans", temperature_f=85.0, conditions="partly cloudy", humidity_pct=66.0, wind_mph=5.0),
    "nashville": WeatherResult(city="Nashville", temperature_f=53.0, conditions="partly cloudy", humidity_pct=64.0, wind_mph=14.0),
    "philadelphia": WeatherResult(city="Philadelphia", temperature_f=52.0, conditions="clear", humidity_pct=65.0, wind_mph=10.0),
    "washington": WeatherResult(city="Washington", temperature_f=62.0, conditions="rainy", humidity_pct=59.0, wind_mph=6.0),
    "minneapolis": WeatherResult(city="Minneapolis", temperature_f=29.0, conditions="windy", humidity_pct=81.0, wind_mph=17.0),
    "winnipeg": WeatherResult(city="Winnipeg", temperature_f=34.0, conditions="windy", humidity_pct=76.0, wind_mph=10.0),
    "lyon": WeatherResult(city="Lyon", temperature_f=72.0, conditions="sunny", humidity_pct=66.0, wind_mph=13.0),
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
