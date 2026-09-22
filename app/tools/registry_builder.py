from app.tools.builtin.add_days import AddDaysTool
from app.tools.builtin.aggregate_records import AggregateRecordsTool
from app.tools.builtin.calculator import CalculatorTool
from app.tools.builtin.calendar import CalendarCreateTool, CalendarListTool, CalendarStore
from app.tools.builtin.compound_interest import CompoundInterestTool
from app.tools.builtin.converter import ConverterTool
from app.tools.builtin.date_diff import DateDiffTool
from app.tools.builtin.distance import DistanceTool
from app.tools.builtin.email import SendEmailTool
from app.tools.builtin.encoding import Base64DecodeTool, Base64EncodeTool
from app.tools.builtin.filter_records import FilterRecordsTool
from app.tools.builtin.files import FileReadTool, FileWriteTool, VirtualFileSystem
from app.tools.builtin.percentile import PercentileTool
from app.tools.builtin.prime_checker import PrimeCheckerTool
from app.tools.builtin.roman_numeral import RomanNumeralTool
from app.tools.builtin.scientific_calculator import ScientificCalculatorTool
from app.tools.builtin.stats import StatsTool
from app.tools.builtin.stock_price import StockPriceTool
from app.tools.builtin.temperature import TemperatureConverterTool
from app.tools.builtin.weather import WeatherTool
from app.tools.builtin.weather_forecast import WeatherForecastTool
from app.tools.builtin.word_count import WordCountTool
from app.tools.registry import ToolRegistry


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()  # FinishTool is auto-registered by ToolRegistry itself
    registry.register(CalculatorTool())
    registry.register(ConverterTool())
    registry.register(WeatherTool())
    registry.register(StockPriceTool())
    registry.register(WordCountTool())
    registry.register(PrimeCheckerTool())
    registry.register(RomanNumeralTool())
    registry.register(StatsTool())
    registry.register(DateDiffTool())
    registry.register(TemperatureConverterTool())
    registry.register(Base64EncodeTool())
    registry.register(Base64DecodeTool())
    registry.register(DistanceTool())
    registry.register(CompoundInterestTool())
    registry.register(FilterRecordsTool())
    registry.register(WeatherForecastTool())
    registry.register(ScientificCalculatorTool())
    registry.register(PercentileTool())
    registry.register(AggregateRecordsTool())
    registry.register(AddDaysTool())

    calendar_store = CalendarStore()
    registry.register(CalendarListTool(calendar_store))
    registry.register(CalendarCreateTool(calendar_store))

    fs = VirtualFileSystem()
    registry.register(FileReadTool(fs))
    registry.register(FileWriteTool(fs))

    registry.register(SendEmailTool())

    return registry
