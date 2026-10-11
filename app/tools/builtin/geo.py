import math

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter, ToolPricing

# (latitude, longitude) of every city the weather and distance tools know about. The distance_between lookup table
# is computed from these too, so both distance tools always agree on any pair they both cover.
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "accra": (5.60, -0.19), "addis ababa": (9.03, 38.74), "amsterdam": (52.37, 4.90), "anchorage": (61.22, -149.90),
    "athens": (37.98, 23.73), "atlanta": (33.75, -84.39), "auckland": (-36.85, 174.76), "bangkok": (13.76, 100.50),
    "barcelona": (41.39, 2.17), "beijing": (39.90, 116.41), "berlin": (52.52, 13.40), "bogota": (4.71, -74.07),
    "boston": (42.36, -71.06), "brisbane": (-27.47, 153.03), "brussels": (50.85, 4.35), "budapest": (47.50, 19.04),
    "buenos aires": (-34.60, -58.38), "cairo": (30.04, 31.24), "calgary": (51.05, -114.07), "cape town": (-33.92, 18.42),
    "caracas": (10.48, -66.90), "casablanca": (33.57, -7.59), "chicago": (41.88, -87.63), "colombo": (6.93, 79.86),
    "copenhagen": (55.68, 12.57), "dallas": (32.78, -96.80), "delhi": (28.61, 77.21), "denver": (39.74, -104.99),
    "dhaka": (23.81, 90.41), "doha": (25.29, 51.53), "dubai": (25.20, 55.27), "dublin": (53.35, -6.26),
    "edinburgh": (55.95, -3.19), "hanoi": (21.03, 105.85), "havana": (23.11, -82.37), "helsinki": (60.17, 24.94),
    "hong kong": (22.32, 114.17), "honolulu": (21.31, -157.86), "houston": (29.76, -95.37), "istanbul": (41.01, 28.98),
    "jakarta": (-6.21, 106.85), "johannesburg": (-26.20, 28.05), "karachi": (24.86, 67.01), "kathmandu": (27.72, 85.32),
    "kuala lumpur": (3.14, 101.69), "kyiv": (50.45, 30.52), "lagos": (6.52, 3.38), "las vegas": (36.17, -115.14),
    "lima": (-12.05, -77.04), "lisbon": (38.72, -9.14), "london": (51.51, -0.13), "los angeles": (34.05, -118.24),
    "lyon": (45.76, 4.84), "madrid": (40.42, -3.70), "manila": (14.60, 120.98), "melbourne": (-37.81, 144.96),
    "mexico city": (19.43, -99.13), "miami": (25.76, -80.19), "milan": (45.46, 9.19), "minneapolis": (44.98, -93.27),
    "montreal": (45.50, -73.57), "moscow": (55.76, 37.62), "mumbai": (19.08, 72.88), "munich": (48.14, 11.58),
    "nairobi": (-1.29, 36.82), "nashville": (36.16, -86.78), "new orleans": (29.95, -90.07), "new york": (40.71, -74.01),
    "osaka": (34.69, 135.50), "oslo": (59.91, 10.75), "ottawa": (45.42, -75.70), "paris": (48.86, 2.35),
    "perth": (-31.95, 115.86), "philadelphia": (39.95, -75.17), "phoenix": (33.45, -112.07), "prague": (50.08, 14.44),
    "reykjavik": (64.15, -21.94), "rio de janeiro": (-22.91, -43.17), "riyadh": (24.71, 46.68), "rome": (41.90, 12.50),
    "san francisco": (37.77, -122.42), "santiago": (-33.45, -70.67), "sao paulo": (-23.55, -46.63), "seattle": (47.61, -122.33),
    "seoul": (37.57, 126.98), "shanghai": (31.23, 121.47), "singapore": (1.35, 103.82), "stockholm": (59.33, 18.07),
    "sydney": (-33.87, 151.21), "taipei": (25.03, 121.57), "tehran": (35.69, 51.39), "tel aviv": (32.09, 34.78),
    "tokyo": (35.68, 139.69), "toronto": (43.65, -79.38), "vancouver": (49.28, -123.12), "vienna": (48.21, 16.37),
    "warsaw": (52.23, 21.01), "washington": (38.91, -77.04), "winnipeg": (49.90, -97.14), "zurich": (47.38, 8.54),
}

_EARTH_RADIUS_MILES = 3958.8


def great_circle_miles(city1: str, city2: str) -> float:
    """Great-circle distance between two known cities, rounded to the nearest mile."""
    (lat1, lon1), (lat2, lon2) = CITY_COORDINATES[city1], CITY_COORDINATES[city2]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(round(_EARTH_RADIUS_MILES * 2 * math.asin(math.sqrt(h))))


class GeoDistanceResult(BaseModel):
    city1: str = Field(description="The first city.")
    city2: str = Field(description="The second city.")
    distance_miles: float = Field(description="Great-circle distance between the two cities, in miles.")


class GeoDistanceTool(Tool):
    metadata = ToolMetadata(
        name="geo_distance",
        description="Compute the great-circle distance in miles between any two supported cities from their coordinates. "
                    "Works for every pair of cities that has weather data, not just a fixed list of pairs.",
        pricing=ToolPricing(base=0.05),
        parameters=[
            ToolParameter(name="city1", type="string", description="The first city."),
            ToolParameter(name="city2", type="string", description="The second city."),
        ],
        returns=GeoDistanceResult.__name__,
    )

    def execute(self, city1: str, city2: str) -> GeoDistanceResult:
        key1, key2 = city1.strip().lower(), city2.strip().lower()
        for key, given in ((key1, city1), (key2, city2)):
            if key not in CITY_COORDINATES:
                raise ValueError(f"no coordinates available for '{given}'")
        return GeoDistanceResult(city1=city1, city2=city2, distance_miles=great_circle_miles(key1, key2))
