import math
import random
import statistics as stats_module
from datetime import date, timedelta

from app.environment.state import Action
from app.environment.task import Task
from app.evaluation.benchmark import BenchmarkCase
from app.tools.builtin.add_days import AddDaysTool
from app.tools.builtin.aggregate_records import AggregateRecordsTool
from app.tools.builtin.calculator import CalculatorTool
from app.tools.builtin.compound_interest import CompoundInterestTool
from app.tools.builtin.converter import _CONVERSION_RATES, _rate
from app.tools.builtin.distance import _DISTANCES
from app.tools.builtin.encoding import Base64EncodeTool
from app.tools.builtin.filter_records import FilterRecordsTool
from app.tools.builtin.percentile import compute_percentile
from app.tools.builtin.prime_checker import PrimeCheckerTool
from app.tools.builtin.roman_numeral import RomanNumeralTool
from app.tools.builtin.stock_price import _PRICES
from app.tools.builtin.temperature import convert_temperature
from app.tools.builtin.weather import _WEATHER_DATA
from app.tools.builtin.weather_forecast import WeatherForecastTool

_rng = random.Random(42)


def calculator_cases(n: int) -> list[BenchmarkCase]:
    ops = ["+", "-", "*"]
    cases = []
    for i in range(n):
        a, b = _rng.randint(10, 999), _rng.randint(2, 99)
        expression = f"{a} {_rng.choice(ops)} {b}"
        answer = str(CalculatorTool().execute(expression).result)
        cases.append(BenchmarkCase(
            task=Task(task_id=f"calc_{i:04d}", description=f"What is {expression}?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="calculator", args={"expression": expression}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def converter_cases(n: int) -> list[BenchmarkCase]:
    pairs = list(_CONVERSION_RATES.keys())
    cases = []
    for i in range(n):
        from_unit, to_unit = _rng.choice(pairs)
        value = round(_rng.uniform(1, 500), 2)
        answer = str(round(value * _rate(from_unit, to_unit), 2))
        cases.append(BenchmarkCase(
            task=Task(task_id=f"conv_{i:04d}", description=f"Convert {value} {from_unit} to {to_unit}."),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="converter", args={"value": value, "from_unit": from_unit, "to_unit": to_unit}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def weather_cases(n: int) -> list[BenchmarkCase]:
    cities = list(_WEATHER_DATA.values())
    cases = []
    for i in range(n):
        data = _rng.choice(cities)
        answer = f"{data.temperature_f}F and {data.conditions}"
        cases.append(BenchmarkCase(
            task=Task(task_id=f"weather_{i:04d}", description=f"What's the weather like in {data.city}?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="weather", args={"city": data.city}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def stock_price_cases(n: int) -> list[BenchmarkCase]:
    tickers = list(_PRICES.keys())
    cases = []
    for i in range(n):
        ticker = _rng.choice(tickers)
        answer = f"${_PRICES[ticker]}"
        cases.append(BenchmarkCase(
            task=Task(task_id=f"stock_{i:04d}", description=f"What's the current price of {ticker.upper()} stock?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="stock_price", args={"ticker": ticker}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


_SENTENCE_POOL = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require large amounts of data.",
    "The stock market fluctuated wildly during the announcement.",
    "She walked along the beach collecting seashells at sunset.",
    "The committee will reconvene next Thursday to finalize the budget.",
    "Rainfall this season has been significantly above average.",
    "The chef prepared a seven course meal for the anniversary dinner.",
    "Researchers discovered a new species of frog in the rainforest.",
]


def word_count_cases(n: int) -> list[BenchmarkCase]:
    cases = []
    for i in range(n):
        text = " ".join(_rng.sample(_SENTENCE_POOL, _rng.randint(2, 4)))
        answer = str(len(text.split()))
        cases.append(BenchmarkCase(
            task=Task(task_id=f"wc_{i:04d}", description=f'How many words are in the following text: "{text}"'),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="word_count", args={"text": text}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def prime_checker_cases(n: int) -> list[BenchmarkCase]:
    checker = PrimeCheckerTool()
    cases = []
    for i in range(n):
        number = _rng.randint(1000, 99999)
        answer = "yes" if checker.execute(number).is_prime else "no"
        cases.append(BenchmarkCase(
            task=Task(task_id=f"prime_{i:04d}", description=f"Is {number} a prime number?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="prime_checker", args={"number": number}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def roman_numeral_cases(n: int) -> list[BenchmarkCase]:
    converter = RomanNumeralTool()
    cases = []
    for i in range(n):
        value = _rng.randint(50, 3999)
        answer = converter.execute(value).roman
        cases.append(BenchmarkCase(
            task=Task(task_id=f"roman_{i:04d}", description=f"What is {value} written as a Roman numeral?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="roman_numeral", args={"value": value}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def stats_cases(n: int) -> list[BenchmarkCase]:
    stat_modes = ["mean", "median", "stdev"]
    cases = []
    for i in range(n):
        numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
        stat = _rng.choice(stat_modes)
        answer = str(round(getattr(stats_module, stat)(numbers), 2))
        cases.append(BenchmarkCase(
            task=Task(task_id=f"stats_{i:04d}", description=f"What is the {stat} of this list of numbers: {numbers}?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="statistics", args={"numbers": numbers, "stat": stat}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def date_diff_cases(n: int) -> list[BenchmarkCase]:
    base = date(2024, 1, 1)
    cases = []
    for i in range(n):
        d1 = base + timedelta(days=_rng.randint(0, 700))
        d2 = d1 + timedelta(days=_rng.randint(1, 400))
        answer = str((d2 - d1).days)
        cases.append(BenchmarkCase(
            task=Task(task_id=f"datediff_{i:04d}", description=f"How many days are between {d1.isoformat()} and {d2.isoformat()}?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="date_diff", args={"date1": d1.isoformat(), "date2": d2.isoformat()}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def calendar_list_cases(n: int) -> list[BenchmarkCase]:
    answer = "Team standup at 09:00 and Lunch with Sam at 13:00 on 2024-05-01"
    return [
        BenchmarkCase(
            task=Task(task_id=f"cal_list_{i:04d}", description="What's on my calendar right now?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="calendar_list", args={}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )
        for i in range(n)
    ]


_EVENT_TITLES = ["Doctor appointment", "Project review", "Client call", "Gym session", "Book club", "Dentist visit"]


def calendar_create_cases(n: int) -> list[BenchmarkCase]:
    cases = []
    for i in range(n):
        title = _rng.choice(_EVENT_TITLES)
        time_str = f"2024-05-02 {_rng.randint(8, 18):02d}:00"
        answer = f"Scheduled '{title}' at {time_str}"
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"cal_create_{i:04d}",
                description=f"Schedule an event called '{title}' at {time_str}, then confirm it's on the calendar.",
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="calendar_create", args={"time": time_str, "title": title}),
                Action(tool_name="calendar_list", args={}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


_NOTE_CONTENTS = ["Buy milk and eggs", "Call the plumber tomorrow", "Finish the quarterly report", "Renew passport before June"]


def file_write_read_cases(n: int) -> list[BenchmarkCase]:
    cases = []
    for i in range(n):
        content = _rng.choice(_NOTE_CONTENTS)
        path = f"note_{i}.txt"
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"file_wr_{i:04d}",
                description=f"Write '{content}' to a file called '{path}', then read it back to confirm.",
            ),
            expected_answer=content,
            expected_actions=[
                Action(tool_name="file_write", args={"path": path, "content": content}),
                Action(tool_name="file_read", args={"path": path}),
                Action(tool_name="finish", args={"answer": content}),
            ],
        ))
    return cases


def file_read_cases(n: int) -> list[BenchmarkCase]:
    answer = "Welcome! This is a scratch filesystem for notes."
    return [
        BenchmarkCase(
            task=Task(task_id=f"file_read_{i:04d}", description="What does readme.txt say?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="file_read", args={"path": "readme.txt"}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )
        for i in range(n)
    ]


_EMAIL_SUBJECTS = ["Meeting follow-up", "Project update", "Quick question", "Invoice attached"]


def email_cases(n: int) -> list[BenchmarkCase]:
    cases = []
    for i in range(n):
        to = f"user{i}@example.com"
        subject = _rng.choice(_EMAIL_SUBJECTS)
        body = "Please let me know if you have any questions."
        answer = f"Email sent to {to}"
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"email_{i:04d}",
                description=f"Send an email to {to} with subject '{subject}' and body '{body}'.",
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="send_email", args={"to": to, "subject": subject, "body": body}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def convert_then_prime_cases(n: int) -> list[BenchmarkCase]:
    checker = PrimeCheckerTool()
    pairs = list(_CONVERSION_RATES.keys())
    cases = []
    for i in range(n):
        from_unit, to_unit = _rng.choice(pairs)
        value = _rng.randint(1, 200)
        converted = round(value * _rate(from_unit, to_unit))
        answer = "yes" if checker.execute(converted).is_prime else "no"
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"conv_prime_{i:04d}",
                description=f"Convert {value} {from_unit} to {to_unit}, round to the nearest whole number, and tell me if that number is prime.",
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="converter", args={"value": value, "from_unit": from_unit, "to_unit": to_unit}),
                Action(tool_name="prime_checker", args={"number": converted}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def convert_sum_prime_cases(n: int) -> list[BenchmarkCase]:
    """3 distinct tools: converter (x2), calculator, prime_checker."""
    checker = PrimeCheckerTool()
    pairs = list(_CONVERSION_RATES.keys())
    cases = []
    for i in range(n):
        (from1, to1), (from2, to2) = _rng.sample(pairs, 2)
        v1, v2 = _rng.randint(1, 200), _rng.randint(1, 200)
        c1 = round(v1 * _rate(from1, to1))
        c2 = round(v2 * _rate(from2, to2))
        total = c1 + c2
        answer = "yes" if checker.execute(total).is_prime else "no"
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"conv_sum_prime_{i:04d}",
                description=(
                    f"Convert {v1} {from1} to {to1} and {v2} {from2} to {to2} (both rounded to the nearest whole "
                    "number), add the two results together, and tell me if the sum is prime."
                ),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="converter", args={"value": v1, "from_unit": from1, "to_unit": to1}),
                Action(tool_name="converter", args={"value": v2, "from_unit": from2, "to_unit": to2}),
                Action(tool_name="calculator", args={"expression": f"{c1} + {c2}"}),
                Action(tool_name="prime_checker", args={"number": total}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def stats_prime_roman_cases(n: int) -> list[BenchmarkCase]:
    """3 distinct tools: statistics, prime_checker, roman_numeral."""
    checker = PrimeCheckerTool()
    converter = RomanNumeralTool()
    cases = []
    for i in range(n):
        numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
        mean_val = round(stats_module.mean(numbers))
        mean_val = max(mean_val, 1)
        is_prime = "yes" if checker.execute(mean_val).is_prime else "no"
        roman = converter.execute(mean_val).roman
        answer = f"mean rounds to {mean_val}, which is {'' if is_prime == 'yes' else 'not '}prime, and is written {roman} in Roman numerals"
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"stats_prime_roman_{i:04d}",
                description=(
                    f"Compute the mean of this list of numbers: {numbers}. Round it to the nearest whole "
                    "number, check whether that rounded number is prime, and convert it to a Roman numeral."
                ),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="statistics", args={"numbers": numbers, "stat": "mean"}),
                Action(tool_name="prime_checker", args={"number": mean_val}),
                Action(tool_name="roman_numeral", args={"value": mean_val}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def weather_stock_email_cases(n: int) -> list[BenchmarkCase]:
    """3 distinct tools: weather, stock_price, send_email."""
    cities = list(_WEATHER_DATA.values())
    tickers = list(_PRICES.keys())
    cases = []
    for i in range(n):
        w = _rng.choice(cities)
        ticker = _rng.choice(tickers)
        price = _PRICES[ticker]
        to = f"user{i}@example.com"
        body = f"Weather in {w.city}: {w.temperature_f}F, {w.conditions}. {ticker.upper()} is trading at ${price}."
        answer = f"Email sent to {to}"
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"weather_stock_email_{i:04d}",
                description=(
                    f"Check the weather in {w.city} and the current price of {ticker.upper()} stock, "
                    f"and email a summary of both to {to}."
                ),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="weather", args={"city": w.city}),
                Action(tool_name="stock_price", args={"ticker": ticker}),
                Action(tool_name="send_email", args={"to": to, "subject": "Daily summary", "body": body}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def weather_compare_cases(n: int) -> list[BenchmarkCase]:
    cities = list(_WEATHER_DATA.values())
    cases = []
    for i in range(n):
        d1, d2 = _rng.sample(cities, 2)
        answer = f"{d1.city if d1.temperature_f > d2.temperature_f else d2.city} is warmer"
        cases.append(BenchmarkCase(
            task=Task(task_id=f"weather_cmp_{i:04d}", description=f"Compare the weather in {d1.city} and {d2.city}. Which is warmer?"),
            expected_answer=answer,
            # NOTE: weather is called twice here with different args. metrics.py's argument_accuracy
            # builds expected_args as a dict keyed by tool_name, so the second `weather` entry below
            # silently overwrites the first -- argument_accuracy will only check the d2 call on this
            # template. task_success (LLM judge) and tool_selection_accuracy are unaffected.
            expected_actions=[
                Action(tool_name="weather", args={"city": d1.city}),
                Action(tool_name="weather", args={"city": d2.city}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def temperature_converter_cases(n: int) -> list[BenchmarkCase]:
    units = ["celsius", "fahrenheit", "kelvin"]
    cases = []
    for i in range(n):
        from_unit, to_unit = _rng.sample(units, 2)
        value = round(_rng.uniform(-40, 100), 1)
        answer = str(round(convert_temperature(value, from_unit, to_unit), 2))
        cases.append(BenchmarkCase(
            task=Task(task_id=f"temp_{i:04d}", description=f"Convert {value} degrees {from_unit} to {to_unit}."),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="temperature_converter", args={"value": value, "from_unit": from_unit, "to_unit": to_unit}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


_ENCODE_TEXTS = ["hello world", "tool use research", "the answer is 42", "meet me at noon", "quarterly report draft"]


def base64_encode_cases(n: int) -> list[BenchmarkCase]:
    cases = []
    for i in range(n):
        text = _rng.choice(_ENCODE_TEXTS)
        answer = Base64EncodeTool().execute(text).encoded
        cases.append(BenchmarkCase(
            task=Task(task_id=f"b64enc_{i:04d}", description=f'Encode the following text as base64: "{text}"'),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="base64_encode", args={"text": text}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def base64_decode_cases(n: int) -> list[BenchmarkCase]:
    cases = []
    for i in range(n):
        original = _rng.choice(_ENCODE_TEXTS)
        encoded = Base64EncodeTool().execute(original).encoded
        cases.append(BenchmarkCase(
            task=Task(task_id=f"b64dec_{i:04d}", description=f'Decode this base64 string: "{encoded}"'),
            expected_answer=original,
            expected_actions=[
                Action(tool_name="base64_decode", args={"text": encoded}),
                Action(tool_name="finish", args={"answer": original}),
            ],
        ))
    return cases


def encode_decode_roundtrip_cases(n: int) -> list[BenchmarkCase]:
    """2 distinct tools, dependency chain: base64_encode -> base64_decode."""
    cases = []
    for i in range(n):
        text = _rng.choice(_ENCODE_TEXTS)
        encoded = Base64EncodeTool().execute(text).encoded
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"b64_roundtrip_{i:04d}",
                description=f'Encode "{text}" as base64, then decode it back to confirm you get the original text.',
            ),
            expected_answer=text,
            expected_actions=[
                Action(tool_name="base64_encode", args={"text": text}),
                Action(tool_name="base64_decode", args={"text": encoded}),
                Action(tool_name="finish", args={"answer": text}),
            ],
        ))
    return cases


def distance_between_cases(n: int) -> list[BenchmarkCase]:
    pairs = list(_DISTANCES.keys())
    cases = []
    for i in range(n):
        pair = _rng.choice(pairs)
        city1, city2 = tuple(pair)
        distance = _DISTANCES[pair]
        answer = f"{distance} miles"
        cases.append(BenchmarkCase(
            task=Task(task_id=f"dist_{i:04d}", description=f"What is the distance between {city1.title()} and {city2.title()}?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="distance_between", args={"city1": city1, "city2": city2}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def compound_interest_cases(n: int) -> list[BenchmarkCase]:
    tool = CompoundInterestTool()
    cases = []
    for i in range(n):
        principal = round(_rng.uniform(100, 10000), 2)
        rate = round(_rng.uniform(1, 10), 1)
        years = _rng.randint(1, 20)
        final = tool.execute(principal, rate, years).final_amount
        answer = str(final)
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"compound_{i:04d}",
                description=f"If I invest ${principal} at {rate}% annual interest compounded yearly for {years} years, what's the final amount?",
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="compound_interest", args={"principal": principal, "rate_pct": rate, "years": years}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


_NAME_POOL = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]


def filter_records_cases(n: int) -> list[BenchmarkCase]:
    tool = FilterRecordsTool()
    cases = []
    for i in range(n):
        names = _rng.sample(_NAME_POOL, _rng.randint(4, 8))
        records = [{"name": name, "score": round(_rng.uniform(0, 100), 1)} for name in names]
        threshold = round(_rng.uniform(30, 80), 1)
        matching_names = sorted(r["name"] for r in tool.execute(records, threshold).matching)
        answer = ", ".join(matching_names) if matching_names else "none"
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"filter_{i:04d}",
                description=f"Given these records: {records}, which names have a score of at least {threshold}?",
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="filter_records", args={"records": records, "threshold": threshold}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def weather_forecast_cases(n: int) -> list[BenchmarkCase]:
    tool = WeatherForecastTool()
    cities = list(_WEATHER_DATA.values())
    cases = []
    for i in range(n):
        city = _rng.choice(cities).city
        days = _rng.randint(2, 5)
        forecast = tool.execute(city, days).forecast
        answer = "; ".join(f"day {d.day}: {d.temperature_f}F {d.conditions}" for d in forecast)
        cases.append(BenchmarkCase(
            task=Task(task_id=f"forecast_{i:04d}", description=f"Give me a {days}-day weather forecast for {city}."),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="weather_forecast", args={"city": city, "days": days}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def scientific_calculator_cases(n: int) -> list[BenchmarkCase]:
    templates = [
        ("sqrt({n})", lambda n: math.sqrt(n)),
        ("sin({n})", lambda n: math.sin(n)),
        ("cos({n})", lambda n: math.cos(n)),
        ("log({n})", lambda n: math.log(n)),
        ("exp({n})", lambda n: math.exp(min(n, 10))),
    ]
    cases = []
    for i in range(n):
        expr_template, fn = _rng.choice(templates)
        n_val = _rng.randint(1, 100)
        expression = expr_template.format(n=n_val)
        answer = str(round(fn(n_val), 4))
        cases.append(BenchmarkCase(
            task=Task(task_id=f"sci_calc_{i:04d}", description=f"What is {expression}?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="scientific_calculator", args={"expression": expression}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def percentile_cases(n: int) -> list[BenchmarkCase]:
    cases = []
    for i in range(n):
        numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(6, 15))]
        p = _rng.choice([10, 25, 50, 75, 90, 95])
        answer = str(round(compute_percentile(numbers, p), 2))
        cases.append(BenchmarkCase(
            task=Task(task_id=f"pctl_{i:04d}", description=f"What is the {p}th percentile of this list of numbers: {numbers}?"),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="percentile", args={"numbers": numbers, "p": p}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


_CATEGORIES = ["food", "travel", "utilities", "entertainment"]


def aggregate_records_cases(n: int) -> list[BenchmarkCase]:
    tool = AggregateRecordsTool()
    cases = []
    for i in range(n):
        records = [
            {"category": _rng.choice(_CATEGORIES), "value": round(_rng.uniform(5, 200), 2)}
            for _ in range(_rng.randint(6, 12))
        ]
        groups = tool.execute(records).groups
        answer = "; ".join(f"{g.group}: {g.total}" for g in groups)
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"agg_{i:04d}",
                description=f"Given these expense records: {records}, what is the total value per category?",
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="aggregate_records", args={"records": records}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def add_days_cases(n: int) -> list[BenchmarkCase]:
    tool = AddDaysTool()
    base = date(2024, 1, 1)
    cases = []
    for i in range(n):
        start = base + timedelta(days=_rng.randint(0, 700))
        days = _rng.randint(-90, 90)
        answer = tool.execute(start.isoformat(), days).result_date
        cases.append(BenchmarkCase(
            task=Task(
                task_id=f"add_days_{i:04d}",
                description=f"What date is {abs(days)} days {'after' if days >= 0 else 'before'} {start.isoformat()}?",
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="add_days", args={"start_date": start.isoformat(), "days": days}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        ))
    return cases


def _weather_lookup():
    d = _rng.choice(list(_WEATHER_DATA.values()))
    return "weather", {"city": d.city}, f"the weather in {d.city}", f"weather in {d.city} is {d.temperature_f}F and {d.conditions}"


def _stock_lookup():
    ticker = _rng.choice(list(_PRICES.keys()))
    return "stock_price", {"ticker": ticker}, f"the price of {ticker.upper()} stock", f"{ticker.upper()} stock is ${_PRICES[ticker]}"


def _prime_lookup():
    number = _rng.randint(1000, 99999)
    is_prime = PrimeCheckerTool().execute(number).is_prime
    return "prime_checker", {"number": number}, f"whether {number} is prime", f"{number} is {'prime' if is_prime else 'not prime'}"


def _roman_lookup():
    value = _rng.randint(50, 3999)
    roman = RomanNumeralTool().execute(value).roman
    return "roman_numeral", {"value": value}, f"the Roman numeral for {value}", f"{value} in Roman numerals is {roman}"


def _word_count_lookup():
    text = " ".join(_rng.sample(_SENTENCE_POOL, _rng.randint(2, 3)))
    return "word_count", {"text": text}, f'how many words are in "{text}"', f"the text has {len(text.split())} words"


def _date_diff_lookup():
    base = date(2024, 1, 1)
    d1 = base + timedelta(days=_rng.randint(0, 700))
    d2 = d1 + timedelta(days=_rng.randint(1, 400))
    days = (d2 - d1).days
    question = f"how many days are between {d1.isoformat()} and {d2.isoformat()}"
    return "date_diff", {"date1": d1.isoformat(), "date2": d2.isoformat()}, question, f"there are {days} days between those dates"


def _converter_lookup():
    pairs = list(_CONVERSION_RATES.keys())
    from_unit, to_unit = _rng.choice(pairs)
    value = round(_rng.uniform(1, 500), 2)
    converted = round(value * _rate(from_unit, to_unit), 2)
    question = f"what {value} {from_unit} is in {to_unit}"
    return "converter", {"value": value, "from_unit": from_unit, "to_unit": to_unit}, question, f"{value} {from_unit} is {converted} {to_unit}"


def _temperature_lookup():
    units = ["celsius", "fahrenheit", "kelvin"]
    from_unit, to_unit = _rng.sample(units, 2)
    value = round(_rng.uniform(-40, 100), 1)
    converted = round(convert_temperature(value, from_unit, to_unit), 2)
    question = f"what {value} degrees {from_unit} is in {to_unit}"
    return "temperature_converter", {"value": value, "from_unit": from_unit, "to_unit": to_unit}, question, f"{value} {from_unit} is {converted} {to_unit}"


def _stats_lookup():
    numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
    stat = _rng.choice(["mean", "median", "stdev"])
    value = round(getattr(stats_module, stat)(numbers), 2)
    question = f"the {stat} of {numbers}"
    return "statistics", {"numbers": numbers, "stat": stat}, question, f"the {stat} of that list is {value}"


def _base64_encode_lookup():
    text = _rng.choice(_ENCODE_TEXTS)
    encoded = Base64EncodeTool().execute(text).encoded
    question = f'the base64 encoding of "{text}"'
    return "base64_encode", {"text": text}, question, f'"{text}" encodes to {encoded}'


def _distance_lookup():
    pair = _rng.choice(list(_DISTANCES.keys()))
    city1, city2 = tuple(pair)
    distance = _DISTANCES[pair]
    question = f"the distance between {city1.title()} and {city2.title()}"
    return "distance_between", {"city1": city1, "city2": city2}, question, f"the distance is {distance} miles"


def _compound_interest_lookup():
    tool = CompoundInterestTool()
    principal = round(_rng.uniform(100, 10000), 2)
    rate = round(_rng.uniform(1, 10), 1)
    years = _rng.randint(1, 20)
    final = tool.execute(principal, rate, years).final_amount
    question = f"the final amount after investing ${principal} at {rate}% for {years} years compounded annually"
    return "compound_interest", {"principal": principal, "rate_pct": rate, "years": years}, question, f"the final amount is {final}"


def _percentile_lookup():
    numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
    p = _rng.choice([10, 25, 50, 75, 90, 95])
    value = round(compute_percentile(numbers, p), 2)
    question = f"the {p}th percentile of {numbers}"
    return "percentile", {"numbers": numbers, "p": p}, question, f"the {p}th percentile is {value}"


def _add_days_lookup():
    base = date(2024, 1, 1)
    start = base + timedelta(days=_rng.randint(0, 700))
    days = _rng.randint(-90, 90)
    result = AddDaysTool().execute(start.isoformat(), days).result_date
    question = f"the date {days} days from {start.isoformat()}"
    return "add_days", {"start_date": start.isoformat(), "days": days}, question, f"the resulting date is {result}"


def _scientific_calculator_lookup():
    templates = [
        ("sqrt({n})", lambda n: math.sqrt(n)),
        ("sin({n})", lambda n: math.sin(n)),
        ("log({n})", lambda n: math.log(n)),
    ]
    expr_template, fn = _rng.choice(templates)
    n = _rng.randint(1, 100)
    expression = expr_template.format(n=n)
    value = round(fn(n), 4)
    question = f"what {expression} evaluates to"
    return "scientific_calculator", {"expression": expression}, question, f"{expression} = {value}"


_LOOKUP_GENERATORS = [
    _weather_lookup, _stock_lookup, _prime_lookup, _roman_lookup,
    _word_count_lookup, _date_diff_lookup, _converter_lookup, _temperature_lookup,
    _stats_lookup, _base64_encode_lookup, _distance_lookup, _compound_interest_lookup,
    _percentile_lookup, _add_days_lookup, _scientific_calculator_lookup,
]


def mixed_lookup_cases(n: int, k: int) -> list[BenchmarkCase]:
    """k independent, unrelated single-tool lookups (no data dependency between them) combined
    into one task and one final answer -- distinct from the convert_then_prime-style chains,
    where each step needs the previous step's result."""
    cases = []
    for i in range(n):
        results = [gen() for gen in _rng.sample(_LOOKUP_GENERATORS, k)]
        actions = [Action(tool_name=name, args=args) for name, args, _, _ in results]
        description = "Tell me " + "; and ".join(q for _, _, q, _ in results) + "."
        answer = "; ".join(a for _, _, _, a in results)
        cases.append(BenchmarkCase(
            task=Task(task_id=f"mix{k}_{i:04d}", description=description),
            expected_answer=answer,
            expected_actions=[*actions, Action(tool_name="finish", args={"answer": answer})],
        ))
    return cases


_BASE_COUNTS: dict[str, int] = {
    "calculator": 200, "converter": 150, "weather": 100, "stock_price": 100,
    "word_count": 120, "prime_checker": 120, "roman_numeral": 100,
    "stats": 120, "date_diff": 100, "calendar_list": 30, "file_read": 30,
    "temperature_converter": 120, "base64_encode": 100, "base64_decode": 100,
    "distance_between": 100, "compound_interest": 120, "filter_records": 100,
    "weather_forecast": 100, "scientific_calculator": 120, "percentile": 100,
    "aggregate_records": 100, "add_days": 100,
    "calendar_create": 150, "file_write_read": 150, "email": 150, "convert_then_prime": 150,
    "weather_compare": 100, "encode_decode_roundtrip": 150, "convert_sum_prime": 150,
    "stats_prime_roman": 150, "weather_stock_email": 150,
    "mix_k2": 400, "mix_k3": 400, "mix_k4": 350, "mix_k5": 300,
}


def build_task_bank(scale: float = 0.12333333333276461) -> list[BenchmarkCase]:
    # scale multiplies every template's case count, preserving the proportional mix/spread
    # across templates and distinct-tool-counts. scale=1.0 is the full bank; the current default
    # was binary-searched against a specific total-API-call budget (steps + judge calls summed
    # across the generated cases). Re-tune if the tool set, template counts, or budget change --
    # see the search pattern used to derive it: binary search over scale, computing
    # sum(len(c.expected_actions) for c in build_task_bank(scale=s)) + len(cases) per candidate s.
    #
    # _rng is reseeded here because it's a shared module-level instance every case-generator
    # function pulls from -- without this reset, calling build_task_bank() more than once in
    # the same process (e.g. from a script that calls it twice, or a test loop) would consume
    # further into the same random stream and silently produce a different, non-reproducible
    # case list on the second call. Reseeding makes every call to this function independently
    # deterministic, not just the first one per process.
    _rng.seed(42)
    n = {name: max(1, round(count * scale)) for name, count in _BASE_COUNTS.items()}
    return [
        # single-tool (1 distinct tool)
        *calculator_cases(n["calculator"]),
        *converter_cases(n["converter"]),
        *weather_cases(n["weather"]),
        *stock_price_cases(n["stock_price"]),
        *word_count_cases(n["word_count"]),
        *prime_checker_cases(n["prime_checker"]),
        *roman_numeral_cases(n["roman_numeral"]),
        *stats_cases(n["stats"]),
        *date_diff_cases(n["date_diff"]),
        *calendar_list_cases(n["calendar_list"]),
        *file_read_cases(n["file_read"]),
        *temperature_converter_cases(n["temperature_converter"]),
        *base64_encode_cases(n["base64_encode"]),
        *base64_decode_cases(n["base64_decode"]),
        *distance_between_cases(n["distance_between"]),
        *compound_interest_cases(n["compound_interest"]),
        *filter_records_cases(n["filter_records"]),
        *weather_forecast_cases(n["weather_forecast"]),
        *scientific_calculator_cases(n["scientific_calculator"]),
        *percentile_cases(n["percentile"]),
        *aggregate_records_cases(n["aggregate_records"]),
        *add_days_cases(n["add_days"]),
        # dependency chains (2-4 tools, each step needs the previous step's result)
        *calendar_create_cases(n["calendar_create"]),
        *file_write_read_cases(n["file_write_read"]),
        *email_cases(n["email"]),
        *convert_then_prime_cases(n["convert_then_prime"]),
        *weather_compare_cases(n["weather_compare"]),
        *encode_decode_roundtrip_cases(n["encode_decode_roundtrip"]),
        *convert_sum_prime_cases(n["convert_sum_prime"]),
        *stats_prime_roman_cases(n["stats_prime_roman"]),
        *weather_stock_email_cases(n["weather_stock_email"]),
        # independent-lookup mixes (no dependency between the tool calls)
        *mixed_lookup_cases(n["mix_k2"], k=2),
        *mixed_lookup_cases(n["mix_k3"], k=3),
        *mixed_lookup_cases(n["mix_k4"], k=4),
        *mixed_lookup_cases(n["mix_k5"], k=5),
    ]
