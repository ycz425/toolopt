from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter, ToolPricing
from app.tools.builtin.geo import great_circle_miles


class DistanceResult(BaseModel):
    city1: str = Field(description="The first city.")
    city2: str = Field(description="The second city.")
    distance_miles: float = Field(description="Approximate distance between the two cities, in miles.")


# City pairs the lookup table covers. Their distances come from the shared city coordinates in geo.py, so
# distance_between and geo_distance always agree on any pair both of them cover.
_DISTANCE_PAIRS: list[tuple[str, str]] = [
    ("toronto", "vancouver"), ("toronto", "new york"), ("london", "paris"), ("london", "berlin"),
    ("new york", "london"), ("tokyo", "seoul"), ("sydney", "singapore"), ("dubai", "cairo"),
    ("mumbai", "singapore"), ("rome", "amsterdam"), ("mexico city", "new york"), ("rio de janeiro", "mexico city"),
    ("moscow", "berlin"), ("bangkok", "singapore"), ("chicago", "new york"), ("new york", "hong kong"),
    ("lyon", "boston"), ("addis ababa", "toronto"), ("lyon", "amsterdam"), ("seattle", "rome"),
    ("jakarta", "lima"), ("las vegas", "rio de janeiro"), ("tel aviv", "berlin"), ("helsinki", "melbourne"),
    ("kathmandu", "colombo"), ("tel aviv", "kuala lumpur"), ("lyon", "cairo"), ("calgary", "miami"),
    ("lisbon", "paris"), ("winnipeg", "moscow"), ("karachi", "taipei"), ("tel aviv", "london"),
    ("minneapolis", "chicago"), ("hong kong", "helsinki"), ("accra", "karachi"), ("addis ababa", "dhaka"),
    ("houston", "perth"), ("lisbon", "taipei"), ("dhaka", "tehran"), ("kuala lumpur", "karachi"),
    ("calgary", "brisbane"), ("kathmandu", "madrid"), ("tel aviv", "houston"), ("taipei", "bangkok"),
    ("kyiv", "rio de janeiro"), ("munich", "hong kong"), ("copenhagen", "mumbai"), ("havana", "montreal"),
    ("beijing", "mumbai"), ("denver", "havana"), ("oslo", "rio de janeiro"), ("anchorage", "buenos aires"),
    ("caracas", "athens"), ("rome", "ottawa"), ("bangkok", "manila"), ("phoenix", "philadelphia"),
    ("moscow", "munich"), ("hanoi", "los angeles"), ("havana", "phoenix"), ("los angeles", "honolulu"),
    ("shanghai", "dhaka"), ("milan", "prague"), ("kyiv", "houston"), ("budapest", "copenhagen"),
    ("dubai", "las vegas"), ("athens", "new york"), ("prague", "doha"), ("osaka", "hong kong"),
    ("honolulu", "new york"), ("brussels", "vienna"), ("kathmandu", "lima"), ("edinburgh", "dhaka"),
    ("chicago", "mexico city"), ("atlanta", "cairo"), ("singapore", "madrid"), ("barcelona", "paris"),
    ("barcelona", "washington"), ("seoul", "beijing"), ("auckland", "madrid"), ("milan", "amsterdam"),
    ("lagos", "delhi"), ("brisbane", "helsinki"), ("dubai", "lisbon"), ("berlin", "perth"),
    ("boston", "beijing"), ("mumbai", "barcelona"), ("new york", "santiago"), ("dubai", "madrid"),
    ("singapore", "addis ababa"), ("phoenix", "chicago"), ("madrid", "rio de janeiro"), ("osaka", "vancouver"),
    ("kyiv", "barcelona"), ("lima", "seoul"), ("paris", "colombo"), ("honolulu", "montreal"),
    ("mexico city", "los angeles"), ("madrid", "sydney"), ("boston", "houston"), ("stockholm", "bogota"),
]

_DISTANCES: dict[frozenset[str], float] = {frozenset(pair): great_circle_miles(*pair) for pair in _DISTANCE_PAIRS}


class DistanceTool(Tool):
    metadata = ToolMetadata(
        name="distance_between",
        description="Look up the distance in miles between two cities from a table of 100 common city pairs. "
                    "Fails for pairs that aren't in the table.",
        pricing=ToolPricing(base=0.02),
        parameters=[
            ToolParameter(name="city1", type="string", description="The first city."),
            ToolParameter(name="city2", type="string", description="The second city."),
        ],
        returns=DistanceResult.__name__,
    )

    def execute(self, city1: str, city2: str) -> DistanceResult:
        key = frozenset({city1.strip().lower(), city2.strip().lower()})
        distance = _DISTANCES.get(key)
        if distance is None:
            raise ValueError(f"no distance data available between '{city1}' and '{city2}'")
        return DistanceResult(city1=city1, city2=city2, distance_miles=distance)
