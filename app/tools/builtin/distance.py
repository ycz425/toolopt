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
