from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class StockPriceResult(BaseModel):
    ticker: str = Field(description="The stock ticker symbol.")
    price: float = Field(description="The (simulated) current price in USD.")


_PRICES: dict[str, float] = {
    "aapl": 227.50,
    "goog": 168.30,
    "msft": 415.20,
    "amzn": 186.75,
    "tsla": 248.90,
    "nvda": 138.45,
    "meta": 563.20,
    "nflx": 690.10,
    "jpm": 218.60,
    "dis": 112.35,
    "ba": 172.80,
    "ko": 63.15,
    "pg": 168.40,
    "xom": 118.25,
    "wmt": 92.60,
    "v": 315.75,
    "ma": 525.90,
    "unh": 498.20,
    "hd": 402.10,
    "pfe": 26.85,
}


class StockPriceTool(Tool):
    metadata = ToolMetadata(
        name="stock_price",
        description="Look up the current price of a stock by ticker symbol.",
        parameters=[
            ToolParameter(name="ticker", type="string", description="Stock ticker symbol, e.g. 'AAPL'."),
        ],
        returns=StockPriceResult.__name__,
    )

    def execute(self, ticker: str) -> StockPriceResult:
        price = _PRICES.get(ticker.strip().lower())
        if price is None:
            raise ValueError(f"no price data available for ticker '{ticker}'")
        return StockPriceResult(ticker=ticker.upper(), price=price)

    def cost(self, ticker: str) -> float:
        return 0.02
