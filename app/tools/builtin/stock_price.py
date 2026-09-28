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
    "adbe": 196.9,
    "crm": 460.51,
    "orcl": 188.18,
    "intc": 310.05,
    "amd": 123.18,
    "csco": 221.26,
    "ibm": 30.53,
    "qcom": 165.26,
    "txn": 28.9,
    "avgo": 445.19,
    "pypl": 339.61,
    "shop": 129.88,
    "uber": 295.36,
    "abnb": 562.09,
    "spot": 81.64,
    "snap": 494.97,
    "pins": 270.66,
    "zm": 307.1,
    "now": 504.08,
    "intu": 247.99,
    "mu": 313.88,
    "amat": 418.89,
    "lrcx": 589.82,
    "adp": 218.77,
    "bac": 502.73,
    "wfc": 429.9,
    "gs": 388.87,
    "ms": 254.72,
    "axp": 221.58,
    "schw": 51.55,
    "blk": 95.29,
    "jnj": 61.02,
    "mrk": 449.72,
    "abbv": 168.24,
    "lly": 114.68,
    "tmo": 69.0,
    "abt": 507.94,
    "bmy": 524.91,
    "amgn": 408.92,
    "gild": 183.52,
    "cvs": 160.48,
    "pep": 189.97,
    "mcd": 286.48,
    "sbux": 111.37,
    "nke": 278.58,
    "cost": 172.68,
    "tgt": 577.84,
    "low": 584.12,
    "pm": 337.3,
    "mdlz": 161.78,
    "cl": 580.09,
    "kmb": 199.54,
    "cvx": 226.82,
    "cop": 20.62,
    "slb": 241.34,
    "oxy": 295.29,
    "cat": 311.6,
    "de": 136.57,
    "ge": 312.75,
    "hon": 22.87,
    "mmm": 173.22,
    "ups": 72.06,
    "fdx": 251.72,
    "lmt": 44.17,
    "rtx": 33.05,
    "gd": 196.46,
    "nee": 155.03,
    "duk": 359.64,
    "vz": 326.93,
    "tmus": 455.31,
    "cmcsa": 401.38,
    "gm": 435.28,
    "rivn": 529.87,
    "amt": 245.92,
    "pld": 209.16,
    "bkng": 591.14,
    "mar": 106.69,
    "dal": 440.01,
    "ual": 393.07,
    "luv": 45.4,
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
