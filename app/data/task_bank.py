import itertools
import math
import random
import re
import statistics as stats_module
from datetime import date, timedelta
from typing import Callable, NamedTuple

from app.environment.state import Action
from app.environment.task import Task
from app.data.labeled_task import LabeledTask
from app.tools.builtin.add_days import AddDaysTool
from app.tools.builtin.aggregate_records import AggregateRecordsTool
from app.tools.builtin.calculator import CalculatorTool
from app.tools.builtin.compound_interest import CompoundInterestTool
from app.tools.builtin.converter import CURRENCIES, _CONVERSION_RATES, _rate
from app.tools.builtin.distance import _DISTANCES
from app.tools.builtin.email import SHORT_EMAIL_MAX_BODY
from app.tools.builtin.encoding import Base64EncodeTool
from app.tools.builtin.filter_records import FilterRecordsTool
from app.tools.builtin.geo import great_circle_miles
from app.tools.builtin.percentile import compute_percentile
from app.tools.builtin.prime_checker import SMALL_PRIME_LIMIT, PrimeCheckerTool
from app.tools.builtin.roman_numeral import RomanNumeralTool
from app.tools.builtin.stock_price import _PRICES
from app.tools.builtin.temperature import convert_temperature
from app.tools.builtin.weather import _WEATHER_DATA
from app.tools.builtin.weather_forecast import WeatherForecastTool
from app.tools.registry_builder import build_registry

_rng = random.Random(42)


def _unique(n: int, make: Callable[[int], LabeledTask], max_attempts_per_task: int = 100) -> list[LabeledTask]:
    """Calls make(i) until it has n tasks with distinct descriptions, i.e. samples tasks without
    replacement. i is the index of the task being generated (used for task_ids), so a rejected
    duplicate is retried with the same i. Raises instead of padding with duplicates when a
    template's value space is too small for n."""
    labeled_tasks: list[LabeledTask] = []
    seen: set[str] = set()
    attempts = 0
    while len(labeled_tasks) < n:
        attempts += 1
        labeled_task = make(len(labeled_tasks))
        if attempts > max_attempts_per_task * n:
            raise ValueError(
                f"template '{labeled_task.template}' produced only {len(labeled_tasks)} unique tasks "
                f"out of the {n} requested; widen its value pools or lower its count"
            )
        if labeled_task.task.description in seen:
            continue
        seen.add(labeled_task.task.description)
        labeled_tasks.append(labeled_task)
    return labeled_tasks


# ---------------------------------------------------------------------------------------------------------------------
# Paths. A task is built from steps; each step lists the valid ways to do it (cheap and pricey tools from the same cost
# family) and, where relevant, plausible ways that fail (a cheap tool outside its coverage). A task's valid paths are
# every combination of its steps' valid options followed by finish; its invalid paths swap one step of the cheapest
# path for a failing option.
# ---------------------------------------------------------------------------------------------------------------------

_REGISTRY = build_registry()
MAX_VALID_PATHS = 8  # combinations beyond this keep the cheapest, the priciest, and a random sample in between


class _Options(NamedTuple):
    valid: list[list[Action]]
    invalid: list[list[Action]] = []


def _call(tool_name: str, **args) -> Action:
    return Action(tool_name=tool_name, args=args)


def _single(tool_name: str, **args) -> _Options:
    """A step only one tool can do."""
    return _Options([[_call(tool_name, **args)]])


def _actions_cost(actions: list[Action]) -> float:
    return sum(_REGISTRY.get(a.tool_name).cost(**a.args) for a in actions)


def _labeled(template: str, task_id: str, description: str, answer: str, steps: list[_Options]) -> LabeledTask:
    finish = [_call("finish", answer=answer)]
    valid = [[a for option in combo for a in option] + finish for combo in itertools.product(*(s.valid for s in steps))]
    valid.sort(key=_actions_cost)  # stable sort, so equal-cost paths keep a fixed order
    if len(valid) > MAX_VALID_PATHS:
        valid = [valid[0], *_rng.sample(valid[1:-1], MAX_VALID_PATHS - 2), valid[-1]]
        valid.sort(key=_actions_cost)

    cheapest_options = [min(s.valid, key=_actions_cost) for s in steps]
    invalid = [
        [a for j, option in enumerate(cheapest_options) for a in (bad if j == i else option)] + finish
        for i, step in enumerate(steps)
        for bad in step.invalid
    ]
    return LabeledTask(
        template=template,
        task=Task(task_id=task_id, description=description),
        expected_answer=answer,
        expected_actions=valid[0],
        alternative_paths=valid[1:],
        invalid_paths=invalid,
    )


# Cost families: each returns every valid way to do one step (and failing cheap attempts where they apply).

def _arithmetic_options(expression: str) -> _Options:
    return _Options([[_call("calculator", expression=expression)], [_call("scientific_calculator", expression=expression)]])


def _scientific_options(function: str, n: int) -> _Options:
    expression = f"{function}({n})"
    if function == "sqrt":  # the basic calculator can take square roots as a power
        return _Options([[_call("calculator", expression=f"{n} ** 0.5")], [_call("scientific_calculator", expression=expression)]])
    return _Options([[_call("scientific_calculator", expression=expression)]], [[_call("calculator", expression=expression)]])


def _weather_options(cities: list[str]) -> _Options:
    return _Options([[_call("weather", city=c) for c in cities], [_call("weather_batch", cities=cities)]])


def _stock_options(tickers: list[str]) -> _Options:
    return _Options([[_call("stock_price", ticker=t) for t in tickers], [_call("stock_prices", tickers=tickers)]])


def _stats_options(numbers: list[float], stats: list[str]) -> _Options:
    return _Options([[_call("statistics", numbers=numbers, stat=s) for s in stats], [_call("describe_numbers", numbers=numbers)]])


def _prime_options(number: int) -> _Options:
    if number <= SMALL_PRIME_LIMIT:
        return _Options([[_call("small_prime_checker", number=number)], [_call("prime_checker", number=number)]])
    return _Options([[_call("prime_checker", number=number)]], [[_call("small_prime_checker", number=number)]])


def _convert_options(value: float, from_unit: str, to_unit: str) -> _Options:
    args = {"value": value, "from_unit": from_unit, "to_unit": to_unit}
    if from_unit in CURRENCIES and to_unit in CURRENCIES:
        return _Options([[_call("currency_converter", **args)], [_call("converter", **args)]])
    return _Options([[_call("converter", **args)]], [[_call("currency_converter", **args)]])


def _distance_options(city1: str, city2: str) -> _Options:
    if frozenset({city1, city2}) in _DISTANCES:
        return _Options([[_call("distance_between", city1=city1, city2=city2)], [_call("geo_distance", city1=city1, city2=city2)]])
    return _Options([[_call("geo_distance", city1=city1, city2=city2)]], [[_call("distance_between", city1=city1, city2=city2)]])


def _file_options(path: str, content: str, verify: bool) -> _Options:
    write = _call("file_write", path=path, content=content)
    plain = [write, _call("file_read", path=path)] if verify else [write]
    return _Options([plain, [_call("file_write_verified", path=path, content=content)]])


def _calendar_options(time: str, title: str, verify: bool) -> _Options:
    create = _call("calendar_create", time=time, title=title)
    plain = [create, _call("calendar_list")] if verify else [create]
    return _Options([plain, [_call("calendar_create_verified", time=time, title=title)]])


def _email_options(to: str, subject: str, body: str) -> _Options:
    args = {"to": to, "subject": subject, "body": body}
    if len(body) <= SHORT_EMAIL_MAX_BODY:
        return _Options([[_call("send_short_email", **args)], [_call("send_email", **args)]])
    return _Options([[_call("send_email", **args)]], [[_call("send_short_email", **args)]])


# ---------------------------------------------------------------------------------------------------------------------
# Shared value pools and phrasing helpers
# ---------------------------------------------------------------------------------------------------------------------

_WORDS = [
    "apple", "river", "meeting", "budget", "coffee", "launch", "draft", "review", "tuesday", "garden",
    "window", "project", "orange", "signal", "harbor", "lemon", "rocket", "silver", "morning", "ticket",
    "planet", "quiet", "forest", "invoice", "summer", "yellow", "bridge", "castle", "pencil", "winter",
    "market", "cloud", "velvet", "anchor", "report", "dinner", "sunset", "copper", "island", "puzzle",
    "travel", "noodle", "falcon", "office", "violet", "canyon", "thunder", "notebook", "picnic", "marble",
]


# Punctuation right after a closing quote whose text already ends in punctuation, e.g. the final "."
# in: body 'Thanks for your help.'.
_DOUBLE_PUNCTUATION_RE = re.compile(r"""([.?!])(["'])[.?!]""")


def _phrase(phrasings: list[str], **fields) -> str:
    return _DOUBLE_PUNCTUATION_RE.sub(r"\1\2", _rng.choice(phrasings).format(**fields))


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _join_names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


# Natural-language names for the statistics tool's `stat` argument.
_STAT_NAMES = {"mean": ["mean", "average"], "median": ["median"], "stdev": ["standard deviation"]}


def _random_text() -> str:
    return " ".join(_rng.sample(_WORDS, _rng.randint(2, 5)))


_FIRST_NAMES = [
    "alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy",
    "kevin", "laura", "mike", "nina", "oscar", "priya", "quinn", "rosa", "sam", "tara",
]
_EMAIL_DOMAINS = ["example.com", "acme.org", "mail.test", "contoso.com", "initech.io"]


def _random_email_address() -> str:
    return f"{_rng.choice(_FIRST_NAMES)}@{_rng.choice(_EMAIL_DOMAINS)}"


def _prime_candidate() -> int:
    """Half the time small enough for small_prime_checker, half the time too big for it."""
    if _rng.random() < 0.5:
        return _rng.randint(1000, SMALL_PRIME_LIMIT)
    return _rng.randint(SMALL_PRIME_LIMIT + 1, 99999)


_CITIES = sorted(_WEATHER_DATA)


def _city_pair() -> tuple[str, str]:
    """Half the time a pair from the distance_between table, half the time a pair only geo_distance can answer."""
    if _rng.random() < 0.5:
        return tuple(_rng.sample(sorted(_rng.choice(list(_DISTANCES))), 2))
    while True:
        city1, city2 = _rng.sample(_CITIES, 2)
        if frozenset({city1, city2}) not in _DISTANCES:
            return city1, city2


_TABLE_PARTNERS: dict[str, list[str]] = {}
for _pair in _DISTANCES:
    _c1, _c2 = sorted(_pair)
    _TABLE_PARTNERS.setdefault(_c1, []).append(_c2)
    _TABLE_PARTNERS.setdefault(_c2, []).append(_c1)


def _next_city(city: str, exclude: set[str]) -> str:
    """A second city for `city`: half the time one it's paired with in the distance_between table (if any), otherwise one
    it isn't (so only geo_distance works)."""
    in_table = [c for c in _TABLE_PARTNERS.get(city, []) if c not in exclude]
    if in_table and _rng.random() < 0.5:
        return _rng.choice(sorted(in_table))
    while True:
        other = _rng.choice(_CITIES)
        if other not in exclude and other != city and frozenset({city, other}) not in _DISTANCES:
            return other


def _pair_distance(city1: str, city2: str) -> float:
    return great_circle_miles(city1, city2)  # the table and geo_distance both use these coordinates


# ---------------------------------------------------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------------------------------------------------

# Phrasings that show the expression symbolically, e.g. "Compute 664 + 16."
_CALC_PHRASINGS = [
    "What is {e}?",
    "Compute {e}.",
    "Calculate {e}.",
    "Evaluate {e}.",
    "Solve {e}.",
    "What does {e} equal?",
    "What's the value of {e}?",
    "Find the result of {e}.",
    "Please calculate {e}.",
    "Can you work out {e} for me?",
    "Could you tell me what {e} comes out to?",
    "I need the result of {e}.",
    "Help me figure out {e}.",
    "Give me the answer to {e}.",
    "Quick math: {e}?",
    "{e} = ?",
    "I'm checking my homework. What is {e}?",
    "Before I send this invoice, can you double-check {e}?",
]

# Phrasings that describe the operation in words, including short word problems, so the model has to
# translate the request into the calculator expression itself. Every phrasing keeps a as the left
# operand, so the expected expression is always "a <op> b".
_CALC_OP_PHRASINGS = {
    "+": [
        "What is {a} plus {b}?",
        "Add {a} and {b}.",
        "What's the sum of {a} and {b}?",
        "What do {a} and {b} add up to?",
        "What's the total of {a} and {b}?",
        "Increase {a} by {b}. What do you get?",
        "A shelf holds {a} books and {b} more arrive. How many books are there now?",
        "I've walked {a} meters and have {b} meters to go. How far is the whole walk?",
    ],
    "-": [
        "What is {a} minus {b}?",
        "Subtract {b} from {a}.",
        "Take {b} away from {a}.",
        "What's {a} less {b}?",
        "Decrease {a} by {b}. What's left?",
        "What's the difference when you subtract {b} from {a}?",
        "I had {a} dollars and spent {b}. How much do I have left?",
        "A tank holds {a} liters and {b} liters drain out. How much remains?",
    ],
    "*": [
        "What is {a} times {b}?",
        "Multiply {a} by {b}.",
        "What's the product of {a} and {b}?",
        "What is {a} multiplied by {b}?",
        "What do you get when you multiply {a} and {b}?",
        "There are {a} boxes with {b} items in each. How many items is that in total?",
        "A ticket costs {a} dollars. How much do {b} tickets cost?",
        "Each of {a} runners ran {b} laps. How many laps were run altogether?",
    ],
}


def calculator_tasks(n: int) -> list[LabeledTask]:
    ops = ["+", "-", "*"]
    def make(i: int) -> LabeledTask:
        a, b = _rng.randint(10, 999), _rng.randint(2, 99)
        op = _rng.choice(ops)
        expression = f"{a} {op} {b}"
        answer = str(CalculatorTool().execute(expression).result)
        if _rng.random() < 0.5:
            description = _rng.choice(_CALC_PHRASINGS).format(e=expression)
        else:
            description = _rng.choice(_CALC_OP_PHRASINGS[op]).format(a=a, b=b)
        return _labeled("calculator", f"calc_{i:04d}", description, answer, [_arithmetic_options(expression)])

    return _unique(n, make)


_CONVERTER_PHRASINGS = [
    "Convert {value} {from_unit} to {to_unit}.",
    "What is {value} {from_unit} in {to_unit}?",
    "How many {to_unit} is {value} {from_unit}?",
    "Express {value} {from_unit} in {to_unit}.",
    "What's the equivalent of {value} {from_unit} in {to_unit}?",
    "Change {value} {from_unit} into {to_unit}.",
    "Can you convert {value} {from_unit} into {to_unit} for me?",
    "I have {value} {from_unit}. How much is that in {to_unit}?",
    "A document lists {value} {from_unit}. What would that be in {to_unit}?",
    "{value} {from_unit} to {to_unit}, please.",
]


def converter_tasks(n: int) -> list[LabeledTask]:
    pairs = list(_CONVERSION_RATES.keys())
    def make(i: int) -> LabeledTask:
        from_unit, to_unit = _rng.choice(pairs)
        value = round(_rng.uniform(1, 500), 2)
        answer = str(round(value * _rate(from_unit, to_unit), 2))
        description = _phrase(_CONVERTER_PHRASINGS, value=value, from_unit=from_unit, to_unit=to_unit)
        return _labeled("converter", f"conv_{i:04d}", description, answer, [_convert_options(value, from_unit, to_unit)])

    return _unique(n, make)


_WEATHER_PHRASINGS = [
    "What's the weather like in {city}?",
    "How's the weather in {city} right now?",
    "What are the current conditions in {city}?",
    "Tell me the current weather in {city}.",
    "Check the weather in {city}.",
    "Can you look up the weather for {city}?",
    "What's the temperature and sky like in {city} at the moment?",
    "I'm heading to {city} today. What's the weather there?",
    "Weather in {city}?",
]


def weather_tasks(n: int) -> list[LabeledTask]:
    cities = list(_WEATHER_DATA.values())
    def make(i: int) -> LabeledTask:
        data = _rng.choice(cities)
        answer = f"{data.temperature_f}F and {data.conditions}"
        description = _phrase(_WEATHER_PHRASINGS, city=data.city)
        return _labeled("weather", f"weather_{i:04d}", description, answer, [_weather_options([data.city])])

    return _unique(n, make)


_WEATHER_MULTI_PHRASINGS = [
    "What's the weather like in {cities}?",
    "Tell me the current weather in {cities}.",
    "Check the weather in {cities}.",
    "How's the weather right now in {cities}?",
    "I'm planning a trip through {cities}. What's the weather in each?",
    "Can you get me the current conditions in {cities}?",
]


def weather_multi_tasks(n: int) -> list[LabeledTask]:
    """2-4 cities: a single weather_batch call is cheaper than one weather call per city."""
    cities = list(_WEATHER_DATA.values())
    def make(i: int) -> LabeledTask:
        picked = _rng.sample(cities, _rng.randint(2, 4))
        answer = "; ".join(f"{w.city}: {w.temperature_f}F and {w.conditions}" for w in picked)
        description = _phrase(_WEATHER_MULTI_PHRASINGS, cities=_join_names([w.city for w in picked]))
        return _labeled("weather_multi", f"weather_multi_{i:04d}", description, answer, [_weather_options([w.city for w in picked])])

    return _unique(n, make)


_STOCK_PHRASINGS = [
    "What's the current price of {ticker} stock?",
    "How much is {ticker} trading at?",
    "Look up the share price for {ticker}.",
    "What's {ticker} going for right now?",
    "Get me the latest price of {ticker}.",
    "What is the stock price of ticker {ticker}?",
    "Can you tell me what {ticker} shares are worth today?",
    "I'm thinking of buying {ticker}. What does one share cost right now?",
    "Price check on {ticker}, please.",
    "{ticker} price?",
]


def stock_price_tasks(n: int) -> list[LabeledTask]:
    tickers = list(_PRICES.keys())
    def make(i: int) -> LabeledTask:
        ticker = _rng.choice(tickers)
        answer = f"${_PRICES[ticker]}"
        description = _phrase(_STOCK_PHRASINGS, ticker=ticker.upper())
        return _labeled("stock_price", f"stock_{i:04d}", description, answer, [_stock_options([ticker])])

    return _unique(n, make)


_STOCK_MULTI_PHRASINGS = [
    "What are the current prices of {tickers}?",
    "Look up the share prices for {tickers}.",
    "How much are {tickers} trading at?",
    "Get me the latest prices of {tickers}.",
    "I'm rebalancing my portfolio. What do {tickers} cost per share right now?",
    "Price check on {tickers}, please.",
]


def stock_multi_tasks(n: int) -> list[LabeledTask]:
    """2-4 tickers: a single stock_prices call is cheaper than one stock_price call per ticker."""
    tickers = list(_PRICES.keys())
    def make(i: int) -> LabeledTask:
        picked = _rng.sample(tickers, _rng.randint(2, 4))
        answer = "; ".join(f"{t.upper()}: ${_PRICES[t]}" for t in picked)
        description = _phrase(_STOCK_MULTI_PHRASINGS, tickers=_join_names([t.upper() for t in picked]))
        return _labeled("stock_multi", f"stock_multi_{i:04d}", description, answer, [_stock_options(picked)])

    return _unique(n, make)


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


_WORD_COUNT_PHRASINGS = [
    'How many words are in the following text: "{text}"',
    'Count the words in: "{text}"',
    'What is the word count of "{text}"?',
    'Give me a word count for: "{text}"',
    'Tell me how many words this contains: "{text}"',
    'I need the number of words in this passage: "{text}"',
    'How long is this text in words? "{text}"',
    'My essay excerpt reads "{text}". How many words is that?',
]


def word_count_tasks(n: int) -> list[LabeledTask]:
    def make(i: int) -> LabeledTask:
        text = " ".join(_rng.sample(_SENTENCE_POOL, _rng.randint(2, 4)))
        answer = str(len(text.split()))
        description = _phrase(_WORD_COUNT_PHRASINGS, text=text)
        return _labeled("word_count", f"wc_{i:04d}", description, answer, [_single("word_count", text=text)])

    return _unique(n, make)


_PRIME_PHRASINGS = [
    "Is {number} a prime number?",
    "Is {number} prime?",
    "Check whether {number} is prime.",
    "Find out whether {number} is a prime number.",
    "Verify whether {number} is prime.",
    "Can you tell me if {number} is a prime?",
    "I need to know if {number} is prime.",
    "Is the number {number} prime? Answer yes or no.",
    "Quick check: is {number} prime?",
]


def prime_checker_tasks(n: int) -> list[LabeledTask]:
    checker = PrimeCheckerTool()
    def make(i: int) -> LabeledTask:
        number = _prime_candidate()
        answer = "yes" if checker.execute(number).is_prime else "no"
        description = _phrase(_PRIME_PHRASINGS, number=number)
        return _labeled("prime_checker", f"prime_{i:04d}", description, answer, [_prime_options(number)])

    return _unique(n, make)


_ROMAN_PHRASINGS = [
    "What is {value} written as a Roman numeral?",
    "Convert {value} to Roman numerals.",
    "Write {value} as a Roman numeral.",
    "What's the Roman numeral for {value}?",
    "How do you write {value} in Roman numerals?",
    "Express {value} in Roman numerals.",
    "Give me the Roman numeral form of {value}.",
    "I'm engraving the number {value} on a plaque. How is it written in Roman numerals?",
    "{value} in Roman numerals?",
]


def roman_numeral_tasks(n: int) -> list[LabeledTask]:
    converter = RomanNumeralTool()
    def make(i: int) -> LabeledTask:
        value = _rng.randint(50, 3999)
        answer = converter.execute(value).roman
        description = _phrase(_ROMAN_PHRASINGS, value=value)
        return _labeled("roman_numeral", f"roman_{i:04d}", description, answer, [_single("roman_numeral", value=value)])

    return _unique(n, make)


_STATS_PHRASINGS = [
    "What is the {stat} of this list of numbers: {numbers}?",
    "Compute the {stat} of {numbers}.",
    "Calculate the {stat} for {numbers}.",
    "Find the {stat} of these values: {numbers}.",
    "Given the data {numbers}, what is the {stat}?",
    "Here are some measurements: {numbers}. What's their {stat}?",
    "I recorded these scores: {numbers}. Can you work out the {stat}?",
    "What's the {stat} of {numbers}?",
]


def stats_tasks(n: int) -> list[LabeledTask]:
    stat_modes = ["mean", "median", "stdev"]
    def make(i: int) -> LabeledTask:
        numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
        stat = _rng.choice(stat_modes)
        answer = str(round(getattr(stats_module, stat)(numbers), 2))
        description = _phrase(_STATS_PHRASINGS, stat=_rng.choice(_STAT_NAMES[stat]), numbers=numbers)
        return _labeled("stats", f"stats_{i:04d}", description, answer, [_stats_options(numbers, [stat])])

    return _unique(n, make)


_STATS_MULTI_PHRASINGS = [
    "What are the {stats} of {numbers}?",
    "Compute the {stats} of these values: {numbers}.",
    "For the data {numbers}, find the {stats}.",
    "Here are my test scores: {numbers}. What are their {stats}?",
    "I need the {stats} of {numbers}.",
    "Summarize {numbers} with the {stats}.",
]


def stats_multi_tasks(n: int) -> list[LabeledTask]:
    """2-3 statistics of the same numbers: one describe_numbers call is cheaper than one statistics call each."""
    stat_modes = ["mean", "median", "stdev"]
    def make(i: int) -> LabeledTask:
        numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
        stats = sorted(_rng.sample(stat_modes, _rng.randint(2, 3)), key=stat_modes.index)
        answer = "; ".join(f"{s}: {round(getattr(stats_module, s)(numbers), 2)}" for s in stats)
        description = _phrase(_STATS_MULTI_PHRASINGS, stats=_join_names([_rng.choice(_STAT_NAMES[s]) for s in stats]), numbers=numbers)
        return _labeled("stats_multi", f"stats_multi_{i:04d}", description, answer, [_stats_options(numbers, stats)])

    return _unique(n, make)


_DATE_DIFF_PHRASINGS = [
    "How many days are between {d1} and {d2}?",
    "How many days separate {d1} and {d2}?",
    "Count the days from {d1} to {d2}.",
    "What's the number of days between {d1} and {d2}?",
    "Find the difference in days between {d1} and {d2}.",
    "How far apart, in days, are {d1} and {d2}?",
    "If today is {d1}, how many days until {d2}?",
    "My project started on {d1} and is due on {d2}. How many days apart are those dates?",
]


def date_diff_tasks(n: int) -> list[LabeledTask]:
    base = date(2024, 1, 1)
    def make(i: int) -> LabeledTask:
        d1 = base + timedelta(days=_rng.randint(0, 700))
        d2 = d1 + timedelta(days=_rng.randint(1, 400))
        answer = str((d2 - d1).days)
        description = _phrase(_DATE_DIFF_PHRASINGS, d1=d1.isoformat(), d2=d2.isoformat())
        return _labeled("date_diff", f"datediff_{i:04d}", description, answer,
                        [_single("date_diff", date1=d1.isoformat(), date2=d2.isoformat())])

    return _unique(n, make)


_EVENT_TITLES = [
    "Doctor appointment", "Project review", "Client call", "Gym session", "Book club", "Dentist visit",
    "Team retrospective", "Lunch with Priya", "Budget planning", "Yoga class", "Design sync", "Parent-teacher meeting",
    "Car service", "Interview panel", "Coffee with Oscar", "Quarterly review", "Piano lesson", "Haircut",
    "Vendor demo", "Birthday dinner",
]


def _event_time() -> str:
    return f"2024-05-{_rng.randint(2, 31):02d} {_rng.randint(8, 18):02d}:{_rng.choice(['00', '30'])}"


_CALENDAR_CREATE_PHRASINGS = [
    "Schedule an event called '{title}' at {time}, then confirm it's on the calendar.",
    "Add '{title}' to my calendar for {time} and then check that it shows up.",
    "Put '{title}' on my calendar at {time}, then list my events to make sure it's there.",
    "Create a calendar event '{title}' at {time} and confirm it was added.",
    "Can you book '{title}' for {time}? Afterwards, verify it's on my calendar.",
    "I need '{title}' scheduled at {time}. Please add it and double-check the calendar.",
]


def calendar_create_tasks(n: int) -> list[LabeledTask]:
    """Create and confirm: calendar_create_verified is cheaper than calendar_create + calendar_list."""
    def make(i: int) -> LabeledTask:
        title, time_str = _rng.choice(_EVENT_TITLES), _event_time()
        answer = f"Scheduled '{title}' at {time_str}"
        description = _phrase(_CALENDAR_CREATE_PHRASINGS, title=title, time=time_str)
        return _labeled("calendar_create", f"cal_create_{i:04d}", description, answer, [_calendar_options(time_str, title, verify=True)])

    return _unique(n, make)


_CALENDAR_ADD_PHRASINGS = [
    "Schedule an event called '{title}' at {time}.",
    "Add '{title}' to my calendar for {time}.",
    "Put '{title}' on my calendar at {time}.",
    "Create a calendar event '{title}' at {time}.",
    "Can you book '{title}' for {time}?",
    "Block out {time} on my calendar for '{title}'.",
]


def calendar_add_tasks(n: int) -> list[LabeledTask]:
    """Create without confirming: plain calendar_create is cheaper than calendar_create_verified."""
    def make(i: int) -> LabeledTask:
        title, time_str = _rng.choice(_EVENT_TITLES), _event_time()
        answer = f"Scheduled '{title}' at {time_str}"
        description = _phrase(_CALENDAR_ADD_PHRASINGS, title=title, time=time_str)
        return _labeled("calendar_add", f"cal_add_{i:04d}", description, answer, [_calendar_options(time_str, title, verify=False)])

    return _unique(n, make)


_NOTE_CONTENTS = [
    "Buy milk and eggs", "Call the plumber tomorrow", "Finish the quarterly report", "Renew passport before June",
    "Book flights for the conference", "Water the plants on Friday", "Send the invoice to Acme", "Pick up dry cleaning",
    "Schedule a dentist checkup", "Back up the laptop", "Return library books", "Pay the electricity bill",
    "Order a birthday cake", "Update the project roadmap", "Cancel the gym trial", "Reply to Nina about the offsite",
    "Buy printer ink", "Plan the team lunch", "Fix the leaking tap", "Review the pull request",
]
_FILE_STEMS = [
    "notes", "todo", "reminders", "groceries", "errands", "work", "ideas", "journal",
    "tasks", "shopping", "weekend", "memo", "scratch", "planning", "followups",
]


_FILE_WRITE_READ_PHRASINGS = [
    "Write '{content}' to a file called '{path}', then read it back to confirm.",
    "Save '{content}' in {path} and then read the file to make sure it was saved.",
    "Create {path} containing '{content}', then open it to verify the contents.",
    "Put the text '{content}' into {path}. Read it back afterwards to check.",
    "Store '{content}' in '{path}', then read {path} to double-check.",
    "Can you jot down '{content}' in a file named {path} and confirm it by reading it back?",
]


def file_write_read_tasks(n: int) -> list[LabeledTask]:
    """Write and confirm: file_write_verified is cheaper than file_write + file_read."""
    def make(i: int) -> LabeledTask:
        content, path = _rng.choice(_NOTE_CONTENTS), f"{_rng.choice(_FILE_STEMS)}.txt"
        description = _phrase(_FILE_WRITE_READ_PHRASINGS, content=content, path=path)
        return _labeled("file_write_read", f"file_wr_{i:04d}", description, content, [_file_options(path, content, verify=True)])

    return _unique(n, make)


_FILE_WRITE_PHRASINGS = [
    "Write '{content}' to a file called '{path}'.",
    "Save '{content}' in {path}.",
    "Create {path} containing '{content}'.",
    "Put the text '{content}' into {path}.",
    "Store '{content}' in '{path}'.",
    "Can you jot down '{content}' in a file named {path}?",
]


def file_write_tasks(n: int) -> list[LabeledTask]:
    """Write without confirming: plain file_write is cheaper than file_write_verified."""
    def make(i: int) -> LabeledTask:
        content, path = _rng.choice(_NOTE_CONTENTS), f"{_rng.choice(_FILE_STEMS)}.txt"
        answer = f"Saved '{content}' to {path}"
        description = _phrase(_FILE_WRITE_PHRASINGS, content=content, path=path)
        return _labeled("file_write", f"file_w_{i:04d}", description, answer, [_file_options(path, content, verify=False)])

    return _unique(n, make)


_EMAIL_SUBJECTS = [
    "Meeting follow-up", "Project update", "Quick question", "Invoice attached", "Schedule change",
    "Welcome aboard", "Action items", "Weekly summary", "Contract review", "Travel plans",
]
# Short bodies fit send_short_email; long ones (over its limit) need send_email.
_EMAIL_BODIES = [
    "Please let me know if you have any questions.",
    "Thanks for your help on this.",
    "Can we find time to discuss this week?",
    "The details are below, let me know what you think.",
    "Just a quick reminder about our deadline.",
    "Looking forward to hearing from you.",
    "Thanks for joining the planning session today. I've attached the notes and the open questions we agreed to follow up on.",
    "Quick update on the launch: the design review is done, the remaining bugs are assigned, and we're on track for the date.",
    "Please review the updated contract terms before Friday. The main changes are in the payment schedule and renewal clause.",
    "Confirming our travel plans: flights land at noon on Tuesday, the hotel is booked for three nights, and dinner is Wednesday.",
    "The quarterly numbers are in. Revenue is up slightly, costs are flat, and the full summary is in the shared folder for you.",
    "Reminder that the office is closed on Monday for maintenance. Please take your laptops home and work remotely until Tuesday.",
]


_EMAIL_PHRASINGS = [
    "Send an email to {to} with subject '{subject}' and body '{body}'.",
    "Email {to}. Subject: '{subject}'. Body: '{body}'",
    "Please send {to} a message with the subject line '{subject}' that says '{body}'.",
    "Write to {to} with the subject '{subject}' and the message '{body}'.",
    "Draft and send an email to {to} titled '{subject}' saying '{body}'.",
    "Can you shoot an email over to {to}? Subject '{subject}', body '{body}'.",
]


def email_tasks(n: int) -> list[LabeledTask]:
    def make(i: int) -> LabeledTask:
        to, subject, body = _random_email_address(), _rng.choice(_EMAIL_SUBJECTS), _rng.choice(_EMAIL_BODIES)
        answer = f"Email sent to {to}"
        description = _phrase(_EMAIL_PHRASINGS, to=to, subject=subject, body=body)
        return _labeled("email", f"email_{i:04d}", description, answer, [_email_options(to, subject, body)])

    return _unique(n, make)


_CONVERT_THEN_PRIME_PHRASINGS = [
    "Convert {value} {from_unit} to {to_unit}, round to the nearest whole number, and tell me if that number is prime.",
    "What's {value} {from_unit} in {to_unit}, rounded to the nearest integer? Is that integer prime?",
    "Take {value} {from_unit}, convert it to {to_unit}, round it to a whole number, and check whether the result is prime.",
    "If I convert {value} {from_unit} into {to_unit} and round to the nearest whole number, is the result a prime number?",
    "Round the {to_unit} equivalent of {value} {from_unit} to the nearest whole number. Is it prime?",
    "First convert {value} {from_unit} to {to_unit}. Then round to the nearest integer and check if it's prime.",
]


def convert_then_prime_tasks(n: int) -> list[LabeledTask]:
    checker = PrimeCheckerTool()
    pairs = list(_CONVERSION_RATES.keys())
    def make(i: int) -> LabeledTask:
        from_unit, to_unit = _rng.choice(pairs)
        value = _rng.randint(1, 200)
        converted = round(value * _rate(from_unit, to_unit))
        answer = "yes" if checker.execute(converted).is_prime else "no"
        description = _phrase(_CONVERT_THEN_PRIME_PHRASINGS, value=value, from_unit=from_unit, to_unit=to_unit)
        return _labeled("convert_then_prime", f"conv_prime_{i:04d}", description, answer,
                        [_convert_options(value, from_unit, to_unit), _prime_options(converted)])

    return _unique(n, make)


# Every phrasing names the first conversion first, so the expected calculator call is always "c1 + c2".
_CONVERT_SUM_PRIME_PHRASINGS = [
    "Convert {v1} {from1} to {to1} and {v2} {from2} to {to2} (both rounded to the nearest whole number), add the two results together, and tell me if the sum is prime.",
    "Convert {v1} {from1} to {to1} and {v2} {from2} to {to2}, rounding each to the nearest whole number. Is the sum of the two rounded values prime?",
    "Add the {to1} equivalent of {v1} {from1} to the {to2} equivalent of {v2} {from2}, rounding each conversion to a whole number first. Is the total a prime number?",
    "I have {v1} {from1} and {v2} {from2}. Convert them to {to1} and {to2} respectively, round both to whole numbers, add them, and check whether the sum is prime.",
    "Round {v1} {from1} in {to1} and {v2} {from2} in {to2} to the nearest integers, sum them, and tell me if that sum is prime.",
]


def convert_sum_prime_tasks(n: int) -> list[LabeledTask]:
    """3 distinct tools: converter (x2), calculator, prime_checker."""
    checker = PrimeCheckerTool()
    pairs = list(_CONVERSION_RATES.keys())
    def make(i: int) -> LabeledTask:
        (from1, to1), (from2, to2) = _rng.sample(pairs, 2)
        v1, v2 = _rng.randint(1, 200), _rng.randint(1, 200)
        c1 = round(v1 * _rate(from1, to1))
        c2 = round(v2 * _rate(from2, to2))
        total = c1 + c2
        answer = "yes" if checker.execute(total).is_prime else "no"
        description = _phrase(_CONVERT_SUM_PRIME_PHRASINGS, v1=v1, from1=from1, to1=to1, v2=v2, from2=from2, to2=to2)
        return _labeled("convert_sum_prime", f"conv_sum_prime_{i:04d}", description, answer, [
            _convert_options(v1, from1, to1),
            _convert_options(v2, from2, to2),
            _arithmetic_options(f"{c1} + {c2}"),
            _prime_options(total),
        ])

    return _unique(n, make)


_STATS_PRIME_ROMAN_PHRASINGS = [
    "Compute the mean of this list of numbers: {numbers}. Round it to the nearest whole number, check whether that rounded number is prime, and convert it to a Roman numeral.",
    "Take the average of {numbers}, round it to a whole number, then tell me whether it's prime and how it's written in Roman numerals.",
    "For the values {numbers}: find the mean, round it to the nearest integer, check if that integer is prime, and give its Roman numeral.",
    "Round the mean of {numbers} to the nearest whole number. Is it prime? And what is it in Roman numerals?",
    "I need three things about the rounded mean of {numbers}: its value, whether it's prime, and its Roman numeral form.",
]


def stats_prime_roman_tasks(n: int) -> list[LabeledTask]:
    """3 distinct tools: statistics, prime_checker, roman_numeral."""
    checker = PrimeCheckerTool()
    converter = RomanNumeralTool()
    def make(i: int) -> LabeledTask:
        numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
        mean_val = max(round(stats_module.mean(numbers)), 1)
        is_prime = checker.execute(mean_val).is_prime
        roman = converter.execute(mean_val).roman
        answer = f"mean rounds to {mean_val}, which is {'' if is_prime else 'not '}prime, and is written {roman} in Roman numerals"
        description = _phrase(_STATS_PRIME_ROMAN_PHRASINGS, numbers=numbers)
        return _labeled("stats_prime_roman", f"stats_prime_roman_{i:04d}", description, answer, [
            _stats_options(numbers, ["mean"]),
            _prime_options(mean_val),
            _single("roman_numeral", value=mean_val),
        ])

    return _unique(n, make)


_WEATHER_STOCK_EMAIL_PHRASINGS = [
    "Check the weather in {city} and the current price of {ticker} stock, and email a summary of both to {to}.",
    "Look up the weather in {city} and {ticker}'s stock price, then send {to} an email summarizing both.",
    "Email {to} a quick summary of today's weather in {city} and the current {ticker} share price.",
    "Can you find the current weather in {city} and the price of {ticker}, and send both to {to} by email?",
    "Get the {city} weather and the {ticker} stock price and mail a summary to {to}.",
]


def weather_stock_email_tasks(n: int) -> list[LabeledTask]:
    """3 distinct tools: weather, stock_price, send_email."""
    cities = list(_WEATHER_DATA.values())
    tickers = list(_PRICES.keys())
    def make(i: int) -> LabeledTask:
        w = _rng.choice(cities)
        ticker = _rng.choice(tickers)
        to = _random_email_address()
        body = f"Weather in {w.city}: {w.temperature_f}F, {w.conditions}. {ticker.upper()} is trading at ${_PRICES[ticker]}."
        answer = f"Email sent to {to}"
        description = _phrase(_WEATHER_STOCK_EMAIL_PHRASINGS, city=w.city, ticker=ticker.upper(), to=to)
        return _labeled("weather_stock_email", f"weather_stock_email_{i:04d}", description, answer, [
            _weather_options([w.city]),
            _stock_options([ticker]),
            _email_options(to, "Daily summary", body),
        ])

    return _unique(n, make)


_WEATHER_COMPARE_PHRASINGS = [
    "Compare the weather in {city1} and {city2}. Which is warmer?",
    "Which city is warmer right now, {city1} or {city2}?",
    "Is it warmer in {city1} or in {city2} at the moment?",
    "Check the weather in {city1} and {city2} and tell me which one is warmer.",
    "Between {city1} and {city2}, where is the temperature higher right now?",
    "I can travel to {city1} or {city2} today. Which one is warmer?",
]


def weather_compare_tasks(n: int) -> list[LabeledTask]:
    cities = list(_WEATHER_DATA.values())
    def make(i: int) -> LabeledTask:
        d1, d2 = _rng.sample(cities, 2)
        while d1.temperature_f == d2.temperature_f:
            d1, d2 = _rng.sample(cities, 2)
        answer = f"{d1.city if d1.temperature_f > d2.temperature_f else d2.city} is warmer"
        description = _phrase(_WEATHER_COMPARE_PHRASINGS, city1=d1.city, city2=d2.city)
        # NOTE: one of the valid paths calls weather twice with different args. metrics.py's argument_accuracy
        # builds expected_args as a dict keyed by tool_name, so it only checks the second weather call there.
        return _labeled("weather_compare", f"weather_cmp_{i:04d}", description, answer, [_weather_options([d1.city, d2.city])])

    return _unique(n, make)


_TEMPERATURE_PHRASINGS = [
    "Convert {value} degrees {from_unit} to {to_unit}.",
    "What is {value} degrees {from_unit} in {to_unit}?",
    "Can you convert {value} degrees {from_unit} into {to_unit}?",
    "Express a temperature of {value} {from_unit} in {to_unit}.",
    "If it's {value} degrees {from_unit}, what's that in {to_unit}?",
    "The thermometer reads {value} degrees {from_unit}. What is that in {to_unit}?",
    "{value} {from_unit} to {to_unit}?",
]


def temperature_converter_tasks(n: int) -> list[LabeledTask]:
    units = ["celsius", "fahrenheit", "kelvin"]
    def make(i: int) -> LabeledTask:
        from_unit, to_unit = _rng.sample(units, 2)
        value = round(_rng.uniform(-40, 100), 1)
        answer = str(round(convert_temperature(value, from_unit, to_unit), 2))
        description = _phrase(_TEMPERATURE_PHRASINGS, value=value, from_unit=from_unit, to_unit=to_unit)
        return _labeled("temperature_converter", f"temp_{i:04d}", description, answer,
                        [_single("temperature_converter", value=value, from_unit=from_unit, to_unit=to_unit)])

    return _unique(n, make)


_BASE64_ENCODE_PHRASINGS = [
    'Encode the following text as base64: "{text}"',
    'What is "{text}" in base64?',
    'Base64-encode "{text}".',
    'Convert "{text}" to base64.',
    'Give me the base64 encoding of "{text}".',
    'Can you base64 encode this string: "{text}"?',
    'I need "{text}" encoded in base64 for an API call.',
]


def base64_encode_tasks(n: int) -> list[LabeledTask]:
    def make(i: int) -> LabeledTask:
        text = _random_text()
        answer = Base64EncodeTool().execute(text).encoded
        description = _phrase(_BASE64_ENCODE_PHRASINGS, text=text)
        return _labeled("base64_encode", f"b64enc_{i:04d}", description, answer, [_single("base64_encode", text=text)])

    return _unique(n, make)


_BASE64_DECODE_PHRASINGS = [
    'Decode this base64 string: "{encoded}"',
    'What does the base64 string "{encoded}" decode to?',
    'Base64-decode "{encoded}".',
    'Convert "{encoded}" from base64 back to plain text.',
    'What is the plain text behind the base64 value "{encoded}"?',
    'Can you decode "{encoded}" for me? It\'s base64.',
    'I found "{encoded}" in a log and it looks like base64. What does it say?',
]


def base64_decode_tasks(n: int) -> list[LabeledTask]:
    def make(i: int) -> LabeledTask:
        original = _random_text()
        encoded = Base64EncodeTool().execute(original).encoded
        description = _phrase(_BASE64_DECODE_PHRASINGS, encoded=encoded)
        return _labeled("base64_decode", f"b64dec_{i:04d}", description, original, [_single("base64_decode", text=encoded)])

    return _unique(n, make)


_ROUNDTRIP_PHRASINGS = [
    'Encode "{text}" as base64, then decode it back to confirm you get the original text.',
    'Base64-encode "{text}" and then decode the result to verify it round-trips.',
    'Check that "{text}" survives a base64 round trip: encode it, then decode it.',
    'Convert "{text}" to base64 and back again, and tell me what you end up with.',
    'Encode the string "{text}" in base64, decode the output, and confirm it matches the original.',
]


def encode_decode_roundtrip_tasks(n: int) -> list[LabeledTask]:
    """2 distinct tools, dependency chain: base64_encode -> base64_decode."""
    def make(i: int) -> LabeledTask:
        text = _random_text()
        encoded = Base64EncodeTool().execute(text).encoded
        description = _phrase(_ROUNDTRIP_PHRASINGS, text=text)
        return _labeled("encode_decode_roundtrip", f"b64_roundtrip_{i:04d}", description, text,
                        [_single("base64_encode", text=text), _single("base64_decode", text=encoded)])

    return _unique(n, make)


_DISTANCE_PHRASINGS = [
    "What is the distance between {city1} and {city2}?",
    "How far is {city1} from {city2}?",
    "How many miles apart are {city1} and {city2}?",
    "What's the distance from {city1} to {city2}?",
    "How far apart are {city1} and {city2}?",
    "Look up the distance between {city1} and {city2}.",
    "I'm flying from {city1} to {city2}. Roughly how far is that?",
    "Distance from {city1} to {city2}?",
]


def distance_between_tasks(n: int) -> list[LabeledTask]:
    """Half the pairs are in distance_between's table (cheap lookup); the rest need geo_distance."""
    def make(i: int) -> LabeledTask:
        city1, city2 = _city_pair()
        answer = f"{_pair_distance(city1, city2)} miles"
        description = _phrase(_DISTANCE_PHRASINGS, city1=city1.title(), city2=city2.title())
        return _labeled("distance_between", f"dist_{i:04d}", description, answer, [_distance_options(city1, city2)])

    return _unique(n, make)


_COMPOUND_INTEREST_PHRASINGS = [
    "If I invest ${principal} at {rate}% annual interest compounded yearly for {duration}, what's the final amount?",
    "What will ${principal} grow to after {duration} at {rate}% interest compounded annually?",
    "I put ${principal} in an account paying {rate}% a year, compounded yearly. How much will I have after {duration}?",
    "Calculate the final balance of ${principal} invested for {duration} at an annual rate of {rate}%, compounded once a year.",
    "If ${principal} earns {rate}% per year, compounded annually, what's it worth after {duration}?",
    "Compound ${principal} at {rate}% annually for {duration}. What's the ending amount?",
]


def compound_interest_tasks(n: int) -> list[LabeledTask]:
    tool = CompoundInterestTool()
    def make(i: int) -> LabeledTask:
        principal = round(_rng.uniform(100, 10000), 2)
        rate = round(_rng.uniform(1, 10), 1)
        years = _rng.randint(1, 20)
        answer = str(tool.execute(principal, rate, years).final_amount)
        description = _phrase(_COMPOUND_INTEREST_PHRASINGS, principal=principal, rate=rate, duration=_plural(years, "year"))
        return _labeled("compound_interest", f"compound_{i:04d}", description, answer,
                        [_single("compound_interest", principal=principal, rate_pct=rate, years=years)])

    return _unique(n, make)


_NAME_POOL = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]


_FILTER_RECORDS_PHRASINGS = [
    "Given these records: {records}, which names have a score of at least {threshold}?",
    "Here are some results: {records}. Who scored {threshold} or higher?",
    "From {records}, list the names with a score greater than or equal to {threshold}.",
    "Which people in {records} have scores of {threshold} or above?",
    "Filter these records to the ones scoring at least {threshold} and tell me their names: {records}",
    "Using a passing mark of {threshold}, which of these people passed? {records}",
]


def filter_records_tasks(n: int) -> list[LabeledTask]:
    tool = FilterRecordsTool()
    def make(i: int) -> LabeledTask:
        names = _rng.sample(_NAME_POOL, _rng.randint(4, 8))
        records = [{"name": name, "score": round(_rng.uniform(0, 100), 1)} for name in names]
        threshold = round(_rng.uniform(30, 80), 1)
        matching_names = sorted(r["name"] for r in tool.execute(records, threshold).matching)
        answer = ", ".join(matching_names) if matching_names else "none"
        description = _phrase(_FILTER_RECORDS_PHRASINGS, records=records, threshold=threshold)
        return _labeled("filter_records", f"filter_{i:04d}", description, answer,
                        [_single("filter_records", records=records, threshold=threshold)])

    return _unique(n, make)


_FORECAST_PHRASINGS = [
    "Give me a {days}-day weather forecast for {city}.",
    "What's the forecast for {city} over the next {days} days?",
    "How will the weather look in {city} for the next {days} days?",
    "Show me the weather forecast in {city} for the coming {days} days.",
    "Can you get me the {days}-day outlook for {city}?",
    "I'm visiting {city} for {days} days starting tomorrow. What's the forecast?",
    "Forecast for {city}, {days} days, please.",
]


def weather_forecast_tasks(n: int) -> list[LabeledTask]:
    tool = WeatherForecastTool()
    cities = list(_WEATHER_DATA.values())
    def make(i: int) -> LabeledTask:
        city = _rng.choice(cities).city
        days = _rng.randint(2, 5)
        forecast = tool.execute(city, days).forecast
        answer = "; ".join(f"day {d.day}: {d.temperature_f}F {d.conditions}" for d in forecast)
        description = _phrase(_FORECAST_PHRASINGS, days=days, city=city)
        return _labeled("weather_forecast", f"forecast_{i:04d}", description, answer, [_single("weather_forecast", city=city, days=days)])

    return _unique(n, make)


# (function, implementation, max argument, ways to say it in words). exp's argument stays small so the expression the
# task asks about, the tool's result, and the expected answer all agree.
_SCI_FUNCTIONS = [
    ("sqrt", math.sqrt, 100, ["the square root of {n}"]),
    ("sin", math.sin, 100, ["the sine of {n} radians", "the sine of {n} (in radians)"]),
    ("cos", math.cos, 100, ["the cosine of {n} radians", "the cosine of {n} (in radians)"]),
    ("log", math.log, 100, ["the natural logarithm of {n}", "the natural log of {n}"]),
    ("exp", math.exp, 10, ["e raised to the power of {n}", "e to the {n}"]),
]
_SCI_PHRASINGS = [
    "What is {e}?",
    "Compute {e}.",
    "Evaluate {e}.",
    "Calculate {e}.",
    "Find {e}.",
    "What does {e} come out to?",
    "Can you work out {e} for me?",
    "I need the value of {e} for a physics problem.",
]


def scientific_calculator_tasks(n: int) -> list[LabeledTask]:
    """Square roots also work on the cheap calculator (as n ** 0.5); trig, log, and exp need the scientific one."""
    def make(i: int) -> LabeledTask:
        function, fn, max_n, word_forms = _rng.choice(_SCI_FUNCTIONS)
        n_val = _rng.randint(1, max_n)
        answer = str(round(fn(n_val), 4))
        shown = f"{function}({n_val})" if _rng.random() < 0.5 else _rng.choice(word_forms).format(n=n_val)
        description = _phrase(_SCI_PHRASINGS, e=shown)
        return _labeled("scientific_calculator", f"sci_calc_{i:04d}", description, answer, [_scientific_options(function, n_val)])

    return _unique(n, make)


_PERCENTILE_PHRASINGS = [
    "What is the {p}th percentile of this list of numbers: {numbers}?",
    "Find the {p}th percentile of {numbers}.",
    "Compute the {p}th percentile for these values: {numbers}.",
    "Given {numbers}, what value sits at the {p}th percentile?",
    "Calculate percentile {p} of {numbers}.",
    "Here are response times in milliseconds: {numbers}. What's the p{p}?",
]


def percentile_tasks(n: int) -> list[LabeledTask]:
    def make(i: int) -> LabeledTask:
        numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(6, 15))]
        p = _rng.choice([10, 25, 50, 75, 90, 95])
        answer = str(round(compute_percentile(numbers, p), 2))
        description = _phrase(_PERCENTILE_PHRASINGS, p=p, numbers=numbers)
        return _labeled("percentile", f"pctl_{i:04d}", description, answer, [_single("percentile", numbers=numbers, p=p)])

    return _unique(n, make)


_CATEGORIES = ["food", "travel", "utilities", "entertainment"]


_AGGREGATE_PHRASINGS = [
    "Given these expense records: {records}, what is the total value per category?",
    "Sum these expenses by category: {records}",
    "Here are my expenses: {records}. How much did I spend in each category?",
    "Group {records} by category and total the values.",
    "What's the total per category for these records? {records}",
    "Can you break down these transactions by category, with a total for each? {records}",
]


def aggregate_records_tasks(n: int) -> list[LabeledTask]:
    tool = AggregateRecordsTool()
    def make(i: int) -> LabeledTask:
        records = [
            {"category": _rng.choice(_CATEGORIES), "value": round(_rng.uniform(5, 200), 2)}
            for _ in range(_rng.randint(6, 12))
        ]
        answer = "; ".join(f"{g.group}: {g.total}" for g in tool.execute(records).groups)
        description = _phrase(_AGGREGATE_PHRASINGS, records=records)
        return _labeled("aggregate_records", f"agg_{i:04d}", description, answer, [_single("aggregate_records", records=records)])

    return _unique(n, make)


_ADD_DAYS_AFTER_PHRASINGS = [
    "What date is {days} after {start}?",
    "What's the date {days} from {start}?",
    "Add {days} to {start}.",
    "If today is {start}, what will the date be in {days}?",
    "A package ships on {start} and arrives {days} later. What's the arrival date?",
]
_ADD_DAYS_BEFORE_PHRASINGS = [
    "What date is {days} before {start}?",
    "Subtract {days} from {start}.",
    "What was the date {days} before {start}?",
    "If today is {start}, what was the date {days} ago?",
    "The deadline is {start}. What date is {days} earlier?",
]


def add_days_tasks(n: int) -> list[LabeledTask]:
    tool = AddDaysTool()
    base = date(2024, 1, 1)
    def make(i: int) -> LabeledTask:
        start = base + timedelta(days=_rng.randint(0, 700))
        days = _rng.randint(-90, 90)
        answer = tool.execute(start.isoformat(), days).result_date
        description = _phrase(
            _ADD_DAYS_AFTER_PHRASINGS if days >= 0 else _ADD_DAYS_BEFORE_PHRASINGS,
            days=_plural(abs(days), "day"),
            start=start.isoformat(),
        )
        return _labeled("add_days", f"add_days_{i:04d}", description, answer,
                        [_single("add_days", start_date=start.isoformat(), days=days)])

    return _unique(n, make)


# ---------------------------------------------------------------------------------------------------------------------
# Templates that put the cost families in new contexts: comparisons and aggregates over batch lookups, chains that mix
# two families, and several independent decisions of the same family in one task
# ---------------------------------------------------------------------------------------------------------------------

_WEATHER_WARMEST_PHRASINGS = [
    "Which of {cities} is the warmest right now?",
    "Of {cities}, where is it warmest at the moment?",
    "Check the weather in {cities} and tell me which is warmest.",
    "I want the warmest place today: {cities}. Which one is it?",
    "Which city has the highest temperature right now: {cities}?",
]


def weather_warmest_tasks(n: int) -> list[LabeledTask]:
    """3-4 cities: one weather_batch call is cheaper than one weather call per city."""
    cities = list(_WEATHER_DATA.values())
    def make(i: int) -> LabeledTask:
        while True:
            picked = _rng.sample(cities, _rng.randint(3, 4))
            temps = sorted((w.temperature_f for w in picked), reverse=True)
            if temps[0] != temps[1]:
                break
        warmest = max(picked, key=lambda w: w.temperature_f)
        answer = f"{warmest.city} is the warmest at {warmest.temperature_f}F"
        description = _phrase(_WEATHER_WARMEST_PHRASINGS, cities=_join_names([w.city for w in picked]))
        return _labeled("weather_warmest", f"weather_warmest_{i:04d}", description, answer, [_weather_options([w.city for w in picked])])

    return _unique(n, make)


_WEATHER_RAIN_PHRASINGS = [
    "Is it raining in any of these cities right now: {cities}?",
    "Which of {cities} currently has rain?",
    "I need to know where it's rainy at the moment: {cities}.",
    "Check whether it's raining in {cities}.",
    "Should anyone in {cities} bring an umbrella today? Which cities are rainy?",
]


def weather_rain_check_tasks(n: int) -> list[LabeledTask]:
    """2-4 cities: one weather_batch call is cheaper than one weather call per city."""
    cities = list(_WEATHER_DATA.values())
    def make(i: int) -> LabeledTask:
        picked = _rng.sample(cities, _rng.randint(2, 4))
        rainy = [w.city for w in picked if w.conditions == "rainy"]
        answer = f"rainy in {_join_names(rainy)}" if rainy else "none of them are rainy"
        description = _phrase(_WEATHER_RAIN_PHRASINGS, cities=_join_names([w.city for w in picked]))
        return _labeled("weather_rain_check", f"weather_rain_{i:04d}", description, answer, [_weather_options([w.city for w in picked])])

    return _unique(n, make)


_STOCK_COMPARE_PHRASINGS = [
    "Which is more expensive per share right now: {tickers}?",
    "Of {tickers}, which stock has the highest price?",
    "Compare the share prices of {tickers}. Which costs the most?",
    "Which of {tickers} is trading at the highest price?",
    "I can afford only one share: {tickers}. Which one is priciest?",
]


def stock_compare_tasks(n: int) -> list[LabeledTask]:
    """2-3 tickers: one stock_prices call is cheaper than one stock_price call per ticker."""
    tickers = list(_PRICES.keys())
    def make(i: int) -> LabeledTask:
        while True:
            picked = _rng.sample(tickers, _rng.randint(2, 3))
            prices = sorted((_PRICES[t] for t in picked), reverse=True)
            if prices[0] != prices[1]:
                break
        top = max(picked, key=lambda t: _PRICES[t])
        answer = f"{top.upper()} is the most expensive at ${_PRICES[top]}"
        description = _phrase(_STOCK_COMPARE_PHRASINGS, tickers=_join_names([t.upper() for t in picked]))
        return _labeled("stock_compare", f"stock_cmp_{i:04d}", description, answer, [_stock_options(picked)])

    return _unique(n, make)


_STOCK_AVERAGE_PHRASINGS = [
    "What's the average share price of {tickers}?",
    "Look up {tickers} and tell me the mean of their current prices.",
    "Compute the average current price across {tickers}.",
    "What do {tickers} cost on average per share right now?",
    "I'm tracking {tickers}. What's their average price today?",
]


def stock_average_tasks(n: int) -> list[LabeledTask]:
    """Two decisions: batch the price lookups (2-4 tickers), then a single mean (statistics beats describe_numbers)."""
    tickers = list(_PRICES.keys())
    def make(i: int) -> LabeledTask:
        picked = _rng.sample(tickers, _rng.randint(2, 4))
        prices = [_PRICES[t] for t in picked]
        answer = str(round(stats_module.mean(prices), 2))
        description = _phrase(_STOCK_AVERAGE_PHRASINGS, tickers=_join_names([t.upper() for t in picked]))
        return _labeled("stock_average", f"stock_avg_{i:04d}", description, answer,
                        [_stock_options(picked), _stats_options(prices, ["mean"])])

    return _unique(n, make)


_PORTFOLIO_PHRASINGS = [
    "I own {holdings}. What's the total value of my shares at current prices?",
    "What is a portfolio of {holdings} worth right now?",
    "Price out my holdings: {holdings}. What's the total?",
    "How much would I get if I sold {holdings} at today's prices?",
    "Calculate the current value of {holdings}.",
]


def stock_portfolio_value_tasks(n: int) -> list[LabeledTask]:
    """Two decisions: batch the price lookups (2-3 tickers), then plain arithmetic (calculator beats scientific)."""
    tickers = list(_PRICES.keys())
    def make(i: int) -> LabeledTask:
        picked = _rng.sample(tickers, _rng.randint(2, 3))
        shares = [_rng.randint(2, 20) for _ in picked]
        expression = " + ".join(f"{q} * {_PRICES[t]}" for q, t in zip(shares, picked))
        answer = str(round(CalculatorTool().execute(expression).result, 2))
        holdings = _join_names([f"{q} shares of {t.upper()}" for q, t in zip(shares, picked)])
        description = _phrase(_PORTFOLIO_PHRASINGS, holdings=holdings)
        return _labeled("stock_portfolio_value", f"portfolio_{i:04d}", description, answer,
                        [_stock_options(picked), _arithmetic_options(expression)])

    return _unique(n, make)


_DISTANCE_COMPARE_PHRASINGS = [
    "Which is farther from {origin}: {city1} or {city2}?",
    "Is {city1} or {city2} a longer trip from {origin}?",
    "From {origin}, which is more distant, {city1} or {city2}?",
    "I'm based in {origin}. Which is farther away, {city1} or {city2}?",
    "Compare the distances from {origin} to {city1} and to {city2}. Which is farther?",
]


def distance_compare_tasks(n: int) -> list[LabeledTask]:
    """Two independent distance decisions (table lookup when the pair is listed, geo_distance otherwise)."""
    def make(i: int) -> LabeledTask:
        while True:
            origin = _rng.choice(sorted(_TABLE_PARTNERS))
            city1 = _next_city(origin, {origin})
            city2 = _next_city(origin, {origin, city1})
            d1, d2 = _pair_distance(origin, city1), _pair_distance(origin, city2)
            if d1 != d2:
                break
        farther, distance = (city1, d1) if d1 > d2 else (city2, d2)
        answer = f"{farther.title()} is farther, at {distance} miles"
        description = _phrase(_DISTANCE_COMPARE_PHRASINGS, origin=origin.title(), city1=city1.title(), city2=city2.title())
        return _labeled("distance_compare", f"dist_cmp_{i:04d}", description, answer,
                        [_distance_options(origin, city1), _distance_options(origin, city2)])

    return _unique(n, make)


_TRIP_TOTAL_PHRASINGS = [
    "I'm flying from {a} to {b} and then on to {c}. How many miles is that in total?",
    "What's the total distance of a trip {a} -> {b} -> {c}?",
    "Add up the distances for a route from {a} to {b} to {c}.",
    "How far is it altogether to go from {a} to {b}, then from {b} to {c}?",
    "Plan a trip {a}, {b}, {c}. What's the total mileage?",
]


def distance_trip_total_tasks(n: int) -> list[LabeledTask]:
    """Two distance decisions, then plain addition (calculator beats scientific)."""
    def make(i: int) -> LabeledTask:
        a = _rng.choice(sorted(_TABLE_PARTNERS))
        b = _next_city(a, {a})
        c = _next_city(b, {a, b})
        d1, d2 = _pair_distance(a, b), _pair_distance(b, c)
        answer = f"{d1 + d2} miles"
        description = _phrase(_TRIP_TOTAL_PHRASINGS, a=a.title(), b=b.title(), c=c.title())
        return _labeled("distance_trip_total", f"trip_{i:04d}", description, answer,
                        [_distance_options(a, b), _distance_options(b, c), _arithmetic_options(f"{d1} + {d2}")])

    return _unique(n, make)


_PRIME_MULTI_PHRASINGS = [
    "Which of these numbers are prime: {numbers}?",
    "Check each of {numbers} and tell me which are prime.",
    "Are any of {numbers} prime? Which ones?",
    "Find the primes among {numbers}.",
    "For {numbers}, which are prime numbers?",
]


def prime_multi_tasks(n: int) -> list[LabeledTask]:
    """2-4 independent prime decisions (small_prime_checker when the number fits, prime_checker otherwise)."""
    checker = PrimeCheckerTool()
    def make(i: int) -> LabeledTask:
        numbers = [_prime_candidate() for _ in range(_rng.randint(2, 4))]
        if len(set(numbers)) < len(numbers):
            numbers = sorted(set(numbers))
        primes = [str(x) for x in numbers if checker.execute(x).is_prime]
        answer = f"prime: {_join_names(primes)}" if primes else "none of them are prime"
        description = _phrase(_PRIME_MULTI_PHRASINGS, numbers=_join_names([str(x) for x in numbers]))
        return _labeled("prime_multi", f"prime_multi_{i:04d}", description, answer, [_prime_options(x) for x in numbers])

    return _unique(n, make)


_CURRENCY_EMAIL_PHRASINGS = [
    "Convert {value} {from_unit} to {to_unit} and email the result to {to}.",
    "Work out what {value} {from_unit} is in {to_unit}, then send {to} an email with the amount.",
    "Email {to} the {to_unit} equivalent of {value} {from_unit}.",
    "Can you convert {value} {from_unit} into {to_unit} and let {to} know by email?",
    "Send {to} a quick email saying how much {value} {from_unit} is in {to_unit}.",
]


def currency_email_tasks(n: int) -> list[LabeledTask]:
    """Two decisions: currency_converter beats converter, then (held out from DPO) short vs regular email."""
    currency_pairs = [p for p in _CONVERSION_RATES if p[0] in CURRENCIES and p[1] in CURRENCIES]
    def make(i: int) -> LabeledTask:
        from_unit, to_unit = _rng.choice(currency_pairs)
        if _rng.random() < 0.5:
            from_unit, to_unit = to_unit, from_unit
        value = round(_rng.uniform(10, 2000), 2)
        converted = round(value * _rate(from_unit, to_unit), 2)
        to = _random_email_address()
        body = f"{value} {from_unit.upper()} is {converted} {to_unit.upper()}."
        answer = f"Email sent to {to}"
        description = _phrase(_CURRENCY_EMAIL_PHRASINGS, value=value, from_unit=from_unit, to_unit=to_unit, to=to)
        return _labeled("currency_email", f"currency_email_{i:04d}", description, answer,
                        [_convert_options(value, from_unit, to_unit), _email_options(to, "Currency conversion", body)])

    return _unique(n, make)


_HYPOTENUSE_PHRASINGS = [
    "A right triangle has legs of {a} and {b}. How long is the hypotenuse?",
    "What's the hypotenuse of a right triangle with sides {a} and {b}?",
    "A ladder reaches {b} feet up a wall from {a} feet away. How long is the ladder?",
    "Compute sqrt({a}^2 + {b}^2).",
    "What is the straight-line distance from (0, 0) to ({a}, {b})?",
]


def hypotenuse_tasks(n: int) -> list[LabeledTask]:
    """Square roots of an expression: the basic calculator can do it as a power (cheaper); naming sqrt there fails."""
    def make(i: int) -> LabeledTask:
        a, b = _rng.randint(3, 60), _rng.randint(3, 60)
        answer = str(round(math.hypot(a, b), 4))
        options = _Options(
            [[_call("calculator", expression=f"({a} ** 2 + {b} ** 2) ** 0.5")],
             [_call("scientific_calculator", expression=f"sqrt({a} ** 2 + {b} ** 2)")]],
            [[_call("calculator", expression=f"sqrt({a} ** 2 + {b} ** 2)")]],
        )
        description = _phrase(_HYPOTENUSE_PHRASINGS, a=a, b=b)
        return _labeled("hypotenuse", f"hyp_{i:04d}", description, answer, [options])

    return _unique(n, make)


_STATS_COMPARE_PHRASINGS = [
    "Which list has the higher {stat}: {list1} or {list2}?",
    "Compare the {stat} of {list1} and {list2}. Which is larger?",
    "Group A scored {list1} and group B scored {list2}. Which group has the higher {stat}?",
    "Is the {stat} of {list1} or of {list2} bigger?",
    "Between {list1} and {list2}, which has the greater {stat}?",
]


def stats_compare_tasks(n: int) -> list[LabeledTask]:
    """One statistic per list: a statistics call each beats a describe_numbers call each."""
    stat_modes = ["mean", "median", "stdev"]
    def make(i: int) -> LabeledTask:
        while True:
            list1 = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 9))]
            list2 = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 9))]
            stat = _rng.choice(stat_modes)
            v1, v2 = (round(getattr(stats_module, stat)(x), 2) for x in (list1, list2))
            if v1 != v2:
                break
        answer = f"the {'first' if v1 > v2 else 'second'} list, with a {stat} of {max(v1, v2)}"
        description = _phrase(_STATS_COMPARE_PHRASINGS, stat=_rng.choice(_STAT_NAMES[stat]), list1=list1, list2=list2)
        return _labeled("stats_compare", f"stats_cmp_{i:04d}", description, answer,
                        [_stats_options(list1, [stat]), _stats_options(list2, [stat])])

    return _unique(n, make)


_NOTE_AND_SCHEDULE_PHRASINGS = [
    "Add '{title}' to my calendar at {time} and save a note in {path} saying '{note}'.",
    "Schedule '{title}' for {time}, and write '{note}' to {path}.",
    "Put '{title}' on my calendar at {time}. Also jot down '{note}' in {path}.",
    "Book '{title}' at {time} and store the note '{note}' in {path}.",
    "Create a calendar event '{title}' at {time}, then save '{note}' to a file called {path}.",
]


def note_and_schedule_tasks(n: int) -> list[LabeledTask]:
    """Calendar and file decisions in one task, neither confirmed: the plain tools beat the verified ones."""
    def make(i: int) -> LabeledTask:
        title, time_str = _rng.choice(_EVENT_TITLES), _event_time()
        note, path = _rng.choice(_NOTE_CONTENTS), f"{_rng.choice(_FILE_STEMS)}.txt"
        answer = f"Scheduled '{title}' at {time_str} and saved the note to {path}"
        description = _phrase(_NOTE_AND_SCHEDULE_PHRASINGS, title=title, time=time_str, note=note, path=path)
        return _labeled("note_and_schedule", f"note_sched_{i:04d}", description, answer,
                        [_calendar_options(time_str, title, verify=False), _file_options(path, note, verify=False)])

    return _unique(n, make)


# ---------------------------------------------------------------------------------------------------------------------
# Mixed lookups: each returns (the step's options, how the question is phrased, its part of the final answer)
# ---------------------------------------------------------------------------------------------------------------------

class _Lookup(NamedTuple):
    options: _Options
    question: str
    answer: str


def _weather_lookup() -> _Lookup:
    """Usually one city; sometimes two, so the batch decision also shows up inside mixes."""
    picked = _rng.sample(list(_WEATHER_DATA.values()), 2 if _rng.random() < 0.3 else 1)
    names = _join_names([d.city for d in picked])
    question = _phrase(["the weather in {c}", "the current conditions in {c}", "what the weather is like in {c}"], c=names)
    answer = "; ".join(f"weather in {d.city} is {d.temperature_f}F and {d.conditions}" for d in picked)
    return _Lookup(_weather_options([d.city for d in picked]), question, answer)


def _stock_lookup() -> _Lookup:
    """Usually one ticker; sometimes two, so the batch decision also shows up inside mixes."""
    picked = _rng.sample(list(_PRICES.keys()), 2 if _rng.random() < 0.3 else 1)
    names = _join_names([t.upper() for t in picked])
    question = _phrase(["the price of {t} stock", "what {t} is trading at", "the current {t} share price"], t=names)
    answer = "; ".join(f"{t.upper()} stock is ${_PRICES[t]}" for t in picked)
    return _Lookup(_stock_options(picked), question, answer)


def _prime_lookup() -> _Lookup:
    number = _prime_candidate()
    is_prime = PrimeCheckerTool().execute(number).is_prime
    question = _phrase(["whether {n} is prime", "if {n} is a prime number"], n=number)
    return _Lookup(_prime_options(number), question, f"{number} is {'prime' if is_prime else 'not prime'}")


def _roman_lookup() -> _Lookup:
    value = _rng.randint(50, 3999)
    roman = RomanNumeralTool().execute(value).roman
    question = _phrase(["the Roman numeral for {v}", "how {v} is written in Roman numerals"], v=value)
    return _Lookup(_single("roman_numeral", value=value), question, f"{value} in Roman numerals is {roman}")


def _word_count_lookup() -> _Lookup:
    text = " ".join(_rng.sample(_SENTENCE_POOL, _rng.randint(2, 3)))
    question = _phrase(['how many words are in "{t}"', 'the word count of "{t}"'], t=text)
    return _Lookup(_single("word_count", text=text), question, f"the text has {len(text.split())} words")


def _date_diff_lookup() -> _Lookup:
    base = date(2024, 1, 1)
    d1 = base + timedelta(days=_rng.randint(0, 700))
    d2 = d1 + timedelta(days=_rng.randint(1, 400))
    question = _phrase(
        ["how many days are between {d1} and {d2}", "the number of days from {d1} to {d2}"],
        d1=d1.isoformat(), d2=d2.isoformat(),
    )
    return _Lookup(_single("date_diff", date1=d1.isoformat(), date2=d2.isoformat()), question,
                   f"there are {(d2 - d1).days} days between those dates")


def _converter_lookup() -> _Lookup:
    from_unit, to_unit = _rng.choice(list(_CONVERSION_RATES.keys()))
    value = round(_rng.uniform(1, 500), 2)
    converted = round(value * _rate(from_unit, to_unit), 2)
    question = _phrase(
        ["what {v} {f} is in {t}", "how many {t} are in {v} {f}", "{v} {f} expressed in {t}"],
        v=value, f=from_unit, t=to_unit,
    )
    return _Lookup(_convert_options(value, from_unit, to_unit), question, f"{value} {from_unit} is {converted} {to_unit}")


def _temperature_lookup() -> _Lookup:
    from_unit, to_unit = _rng.sample(["celsius", "fahrenheit", "kelvin"], 2)
    value = round(_rng.uniform(-40, 100), 1)
    converted = round(convert_temperature(value, from_unit, to_unit), 2)
    question = _phrase(["what {v} degrees {f} is in {t}", "{v} degrees {f} converted to {t}"], v=value, f=from_unit, t=to_unit)
    return _Lookup(_single("temperature_converter", value=value, from_unit=from_unit, to_unit=to_unit), question,
                   f"{value} {from_unit} is {converted} {to_unit}")


def _stats_lookup() -> _Lookup:
    numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
    stat = _rng.choice(["mean", "median", "stdev"])
    value = round(getattr(stats_module, stat)(numbers), 2)
    question = f"the {_rng.choice(_STAT_NAMES[stat])} of {numbers}"
    return _Lookup(_stats_options(numbers, [stat]), question, f"the {stat} of that list is {value}")


def _base64_encode_lookup() -> _Lookup:
    text = _random_text()
    encoded = Base64EncodeTool().execute(text).encoded
    question = _phrase(['the base64 encoding of "{t}"', 'what "{t}" looks like in base64'], t=text)
    return _Lookup(_single("base64_encode", text=text), question, f'"{text}" encodes to {encoded}')


def _distance_lookup() -> _Lookup:
    city1, city2 = _city_pair()
    question = _phrase(["the distance between {c1} and {c2}", "how far {c1} is from {c2}"], c1=city1.title(), c2=city2.title())
    return _Lookup(_distance_options(city1, city2), question, f"the distance is {_pair_distance(city1, city2)} miles")


def _compound_interest_lookup() -> _Lookup:
    principal = round(_rng.uniform(100, 10000), 2)
    rate = round(_rng.uniform(1, 10), 1)
    years = _rng.randint(1, 20)
    final = CompoundInterestTool().execute(principal, rate, years).final_amount
    question = _phrase(
        [
            "the final amount after investing ${p} at {r}% for {d} compounded annually",
            "what ${p} grows to in {d} at {r}% compounded yearly",
        ],
        p=principal, r=rate, d=_plural(years, "year"),
    )
    return _Lookup(_single("compound_interest", principal=principal, rate_pct=rate, years=years), question,
                   f"the final amount is {final}")


def _percentile_lookup() -> _Lookup:
    numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
    p = _rng.choice([10, 25, 50, 75, 90, 95])
    value = round(compute_percentile(numbers, p), 2)
    question = _phrase(["the {p}th percentile of {nums}", "the p{p} of {nums}"], p=p, nums=numbers)
    return _Lookup(_single("percentile", numbers=numbers, p=p), question, f"the {p}th percentile is {value}")


def _add_days_lookup() -> _Lookup:
    base = date(2024, 1, 1)
    start = base + timedelta(days=_rng.randint(0, 700))
    days = _rng.randint(-90, 90)
    result = AddDaysTool().execute(start.isoformat(), days).result_date
    question = f"the date {_plural(abs(days), 'day')} {'after' if days >= 0 else 'before'} {start.isoformat()}"
    return _Lookup(_single("add_days", start_date=start.isoformat(), days=days), question, f"the resulting date is {result}")


def _scientific_calculator_lookup() -> _Lookup:
    function, fn, max_n, word_forms = _rng.choice(_SCI_FUNCTIONS)
    n = _rng.randint(1, max_n)
    expression = f"{function}({n})"
    question = _rng.choice([f"what {expression} evaluates to", word_forms[0].format(n=n)])
    return _Lookup(_scientific_options(function, n), question, f"{expression} = {round(fn(n), 4)}")


_LOOKUP_GENERATORS = [
    _weather_lookup, _stock_lookup, _prime_lookup, _roman_lookup,
    _word_count_lookup, _date_diff_lookup, _converter_lookup, _temperature_lookup,
    _stats_lookup, _base64_encode_lookup, _distance_lookup, _compound_interest_lookup,
    _percentile_lookup, _add_days_lookup, _scientific_calculator_lookup,
]


_MIX_PHRASINGS = [
    "Tell me {questions}.",
    "I need to know {questions}.",
    "Can you find out {questions}?",
    "Please look up {questions}.",
    "Help me with a few things: {questions}.",
    "Quick questions: {questions}.",
]


def mixed_lookup_tasks(n: int, k: int) -> list[LabeledTask]:
    """k independent, unrelated single-tool lookups (no data dependency between them) combined
    into one task and one final answer -- distinct from the convert_then_prime-style chains,
    where each step needs the previous step's result."""
    def make(i: int) -> LabeledTask:
        lookups = [gen() for gen in _rng.sample(_LOOKUP_GENERATORS, k)]
        description = _phrase(_MIX_PHRASINGS, questions="; and ".join(l.question for l in lookups))
        answer = "; ".join(l.answer for l in lookups)
        return _labeled(f"mix_k{k}", f"mix{k}_{i:04d}", description, answer, [l.options for l in lookups])

    return _unique(n, make)


_BASE_COUNTS: dict[str, int] = {
    # single tool, with a cost alternative
    "calculator": 200, "converter": 150, "weather": 100, "stock_price": 100,
    "prime_checker": 120, "stats": 80, "distance_between": 120, "scientific_calculator": 120,
    # single path (no cost alternative): kept for tool coverage in SFT and the benchmark, not for DPO
    "word_count": 80, "roman_numeral": 80, "date_diff": 80, "temperature_converter": 80,
    "base64_encode": 80, "base64_decode": 80, "compound_interest": 80, "filter_records": 80,
    "weather_forecast": 80, "percentile": 80, "aggregate_records": 80, "add_days": 80, "encode_decode_roundtrip": 80,
    # several of the same lookup (batch vs. one call each)
    "weather_multi": 250, "stock_multi": 300, "stats_multi": 550,
    "weather_warmest": 150, "weather_rain_check": 120, "stock_compare": 150,
    # chains and multi-family tasks
    "calendar_create": 200, "calendar_add": 100, "file_write_read": 200, "file_write": 100, "email": 150,
    "convert_then_prime": 150, "weather_compare": 120, "convert_sum_prime": 150, "stats_prime_roman": 150,
    "weather_stock_email": 150, "stock_average": 150, "stock_portfolio_value": 150, "distance_compare": 150,
    "distance_trip_total": 150, "prime_multi": 150, "currency_email": 120, "hypotenuse": 120,
    "stats_compare": 100, "note_and_schedule": 120,
    # independent-lookup mixes
    "mix_k2": 400, "mix_k3": 400, "mix_k4": 350, "mix_k5": 300,
}


def build_task_bank(scale: float = 1.0) -> list[LabeledTask]:
    # scale multiplies every template's task count, preserving the proportional mix/spread
    # across templates and distinct-tool-counts. scale=1.0 is the full bank.
    #
    # _rng is reseeded here because it's a shared module-level instance every task-generator
    # function pulls from -- without this reset, calling build_task_bank() more than once in
    # the same process (e.g. from a script that calls it twice, or a test loop) would consume
    # further into the same random stream and silently produce a different, non-reproducible
    # task list on the second call. Reseeding makes every call to this function independently
    # deterministic, not just the first one per process.
    _rng.seed(42)
    n = {name: max(1, round(count * scale)) for name, count in _BASE_COUNTS.items()}
    return [
        # single-tool (1 distinct tool)
        *calculator_tasks(n["calculator"]),
        *converter_tasks(n["converter"]),
        *weather_tasks(n["weather"]),
        *stock_price_tasks(n["stock_price"]),
        *word_count_tasks(n["word_count"]),
        *prime_checker_tasks(n["prime_checker"]),
        *roman_numeral_tasks(n["roman_numeral"]),
        *stats_tasks(n["stats"]),
        *date_diff_tasks(n["date_diff"]),
        *temperature_converter_tasks(n["temperature_converter"]),
        *base64_encode_tasks(n["base64_encode"]),
        *base64_decode_tasks(n["base64_decode"]),
        *distance_between_tasks(n["distance_between"]),
        *compound_interest_tasks(n["compound_interest"]),
        *filter_records_tasks(n["filter_records"]),
        *weather_forecast_tasks(n["weather_forecast"]),
        *scientific_calculator_tasks(n["scientific_calculator"]),
        *percentile_tasks(n["percentile"]),
        *aggregate_records_tasks(n["aggregate_records"]),
        *add_days_tasks(n["add_days"]),
        # several of the same lookup (batch vs. one call each)
        *weather_multi_tasks(n["weather_multi"]),
        *stock_multi_tasks(n["stock_multi"]),
        *stats_multi_tasks(n["stats_multi"]),
        *weather_warmest_tasks(n["weather_warmest"]),
        *weather_rain_check_tasks(n["weather_rain_check"]),
        *stock_compare_tasks(n["stock_compare"]),
        # chains and multi-family tasks (2-4 tools)
        *calendar_create_tasks(n["calendar_create"]),
        *calendar_add_tasks(n["calendar_add"]),
        *file_write_read_tasks(n["file_write_read"]),
        *file_write_tasks(n["file_write"]),
        *email_tasks(n["email"]),
        *convert_then_prime_tasks(n["convert_then_prime"]),
        *weather_compare_tasks(n["weather_compare"]),
        *encode_decode_roundtrip_tasks(n["encode_decode_roundtrip"]),
        *convert_sum_prime_tasks(n["convert_sum_prime"]),
        *stats_prime_roman_tasks(n["stats_prime_roman"]),
        *weather_stock_email_tasks(n["weather_stock_email"]),
        *stock_average_tasks(n["stock_average"]),
        *stock_portfolio_value_tasks(n["stock_portfolio_value"]),
        *distance_compare_tasks(n["distance_compare"]),
        *distance_trip_total_tasks(n["distance_trip_total"]),
        *prime_multi_tasks(n["prime_multi"]),
        *currency_email_tasks(n["currency_email"]),
        *hypotenuse_tasks(n["hypotenuse"]),
        *stats_compare_tasks(n["stats_compare"]),
        *note_and_schedule_tasks(n["note_and_schedule"]),
        # independent-lookup mixes (no dependency between the tool calls)
        *mixed_lookup_tasks(n["mix_k2"], k=2),
        *mixed_lookup_tasks(n["mix_k3"], k=3),
        *mixed_lookup_tasks(n["mix_k4"], k=4),
        *mixed_lookup_tasks(n["mix_k5"], k=5),
    ]
