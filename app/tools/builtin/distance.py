from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class DistanceResult(BaseModel):
    city1: str = Field(description="The first city.")
    city2: str = Field(description="The second city.")
    distance_miles: float = Field(description="Approximate distance between the two cities, in miles.")


_DISTANCES: dict[frozenset[str], float] = {
    frozenset({"toronto", "vancouver"}): 2075.0,
    frozenset({"toronto", "new york"}): 343.0,
    frozenset({"london", "paris"}): 214.0,
    frozenset({"london", "berlin"}): 578.0,
    frozenset({"new york", "london"}): 3459.0,
    frozenset({"tokyo", "seoul"}): 758.0,
    frozenset({"sydney", "singapore"}): 3915.0,
    frozenset({"dubai", "cairo"}): 1621.0,
    frozenset({"mumbai", "singapore"}): 2427.0,
    frozenset({"rome", "amsterdam"}): 819.0,
    frozenset({"mexico city", "new york"}): 2090.0,
    frozenset({"rio de janeiro", "mexico city"}): 4762.0,
    frozenset({"moscow", "berlin"}): 1006.0,
    frozenset({"bangkok", "singapore"}): 883.0,
    frozenset({"chicago", "new york"}): 713.0,
    frozenset({"new york", "hong kong"}): 8050.0,
    frozenset({"lyon", "boston"}): 3630.0,
    frozenset({"addis ababa", "toronto"}): 7131.0,
    frozenset({"lyon", "amsterdam"}): 457.0,
    frozenset({"seattle", "rome"}): 5665.0,
    frozenset({"jakarta", "lima"}): 11147.0,
    frozenset({"las vegas", "rio de janeiro"}): 6217.0,
    frozenset({"tel aviv", "berlin"}): 1771.0,
    frozenset({"helsinki", "melbourne"}): 9449.0,
    frozenset({"kathmandu", "colombo"}): 1480.0,
    frozenset({"tel aviv", "kuala lumpur"}): 4757.0,
    frozenset({"lyon", "cairo"}): 1789.0,
    frozenset({"calgary", "miami"}): 2494.0,
    frozenset({"lisbon", "paris"}): 903.0,
    frozenset({"winnipeg", "moscow"}): 4688.0,
    frozenset({"karachi", "taipei"}): 3393.0,
    frozenset({"tel aviv", "london"}): 2210.0,
    frozenset({"minneapolis", "chicago"}): 355.0,
    frozenset({"hong kong", "helsinki"}): 4863.0,
    frozenset({"accra", "karachi"}): 4628.0,
    frozenset({"addis ababa", "dhaka"}): 3552.0,
    frozenset({"houston", "perth"}): 10585.0,
    frozenset({"lisbon", "taipei"}): 7001.0,
    frozenset({"dhaka", "tehran"}): 2462.0,
    frozenset({"kuala lumpur", "karachi"}): 2751.0,
    frozenset({"calgary", "brisbane"}): 7791.0,
    frozenset({"kathmandu", "madrid"}): 4958.0,
    frozenset({"tel aviv", "houston"}): 7058.0,
    frozenset({"taipei", "bangkok"}): 1575.0,
    frozenset({"kyiv", "rio de janeiro"}): 6756.0,
    frozenset({"munich", "hong kong"}): 5629.0,
    frozenset({"copenhagen", "mumbai"}): 3989.0,
    frozenset({"havana", "montreal"}): 1624.0,
    frozenset({"beijing", "mumbai"}): 2948.0,
    frozenset({"denver", "havana"}): 1751.0,
    frozenset({"oslo", "rio de janeiro"}): 6475.0,
    frozenset({"anchorage", "buenos aires"}): 8329.0,
    frozenset({"caracas", "athens"}): 5808.0,
    frozenset({"rome", "ottawa"}): 4182.0,
    frozenset({"bangkok", "manila"}): 1373.0,
    frozenset({"phoenix", "philadelphia"}): 2077.0,
    frozenset({"moscow", "munich"}): 1218.0,
    frozenset({"hanoi", "los angeles"}): 7653.0,
    frozenset({"havana", "phoenix"}): 1935.0,
    frozenset({"los angeles", "honolulu"}): 2560.0,
    frozenset({"shanghai", "dhaka"}): 1964.0,
    frozenset({"milan", "prague"}): 401.0,
    frozenset({"kyiv", "houston"}): 5986.0,
    frozenset({"budapest", "copenhagen"}): 629.0,
    frozenset({"dubai", "las vegas"}): 8151.0,
    frozenset({"athens", "new york"}): 4925.0,
    frozenset({"prague", "doha"}): 2610.0,
    frozenset({"osaka", "hong kong"}): 1546.0,
    frozenset({"honolulu", "new york"}): 4959.0,
    frozenset({"brussels", "vienna"}): 568.0,
    frozenset({"kathmandu", "lima"}): 10864.0,
    frozenset({"edinburgh", "dhaka"}): 5003.0,
    frozenset({"chicago", "mexico city"}): 1691.0,
    frozenset({"atlanta", "cairo"}): 6350.0,
    frozenset({"singapore", "madrid"}): 7072.0,
    frozenset({"barcelona", "paris"}): 516.0,
    frozenset({"barcelona", "washington"}): 4032.0,
    frozenset({"seoul", "beijing"}): 592.0,
    frozenset({"auckland", "madrid"}): 12177.0,
    frozenset({"milan", "amsterdam"}): 515.0,
    frozenset({"lagos", "delhi"}): 5024.0,
    frozenset({"brisbane", "helsinki"}): 9138.0,
    frozenset({"dubai", "lisbon"}): 3811.0,
    frozenset({"berlin", "perth"}): 8436.0,
    frozenset({"boston", "beijing"}): 6734.0,
    frozenset({"mumbai", "barcelona"}): 4369.0,
    frozenset({"new york", "santiago"}): 5128.0,
    frozenset({"dubai", "madrid"}): 3514.0,
    frozenset({"singapore", "addis ababa"}): 4504.0,
    frozenset({"phoenix", "chicago"}): 1452.0,
    frozenset({"madrid", "rio de janeiro"}): 5058.0,
    frozenset({"osaka", "vancouver"}): 4906.0,
    frozenset({"kyiv", "barcelona"}): 1488.0,
    frozenset({"lima", "seoul"}): 10133.0,
    frozenset({"paris", "colombo"}): 5291.0,
    frozenset({"honolulu", "montreal"}): 4912.0,
    frozenset({"mexico city", "los angeles"}): 1548.0,
    frozenset({"madrid", "sydney"}): 10989.0,
    frozenset({"boston", "houston"}): 1604.0,
    frozenset({"stockholm", "bogota"}): 6014.0,
}


class DistanceTool(Tool):
    metadata = ToolMetadata(
        name="distance_between",
        description="Look up the approximate distance in miles between two cities.",
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

    def cost(self, city1: str, city2: str) -> float:
        return 0.02
