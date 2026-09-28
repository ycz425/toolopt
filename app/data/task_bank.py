import math
import random
import re
import statistics as stats_module
from datetime import date, timedelta
from typing import Callable

from app.environment.state import Action
from app.environment.task import Task
from app.data.labeled_task import LabeledTask
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
        return LabeledTask(
            template="calculator",
            task=Task(task_id=f"calc_{i:04d}", description=description),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="calculator", args={"expression": expression}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="converter",
            task=Task(task_id=f"conv_{i:04d}", description=_phrase(_CONVERTER_PHRASINGS, value=value, from_unit=from_unit, to_unit=to_unit)),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="converter", args={"value": value, "from_unit": from_unit, "to_unit": to_unit}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="weather",
            task=Task(task_id=f"weather_{i:04d}", description=_phrase(_WEATHER_PHRASINGS, city=data.city)),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="weather", args={"city": data.city}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="stock_price",
            task=Task(task_id=f"stock_{i:04d}", description=_phrase(_STOCK_PHRASINGS, ticker=ticker.upper())),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="stock_price", args={"ticker": ticker}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="word_count",
            task=Task(task_id=f"wc_{i:04d}", description=_phrase(_WORD_COUNT_PHRASINGS, text=text)),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="word_count", args={"text": text}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        number = _rng.randint(1000, 99999)
        answer = "yes" if checker.execute(number).is_prime else "no"
        return LabeledTask(
            template="prime_checker",
            task=Task(task_id=f"prime_{i:04d}", description=_phrase(_PRIME_PHRASINGS, number=number)),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="prime_checker", args={"number": number}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="roman_numeral",
            task=Task(task_id=f"roman_{i:04d}", description=_phrase(_ROMAN_PHRASINGS, value=value)),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="roman_numeral", args={"value": value}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="stats",
            task=Task(task_id=f"stats_{i:04d}", description=_phrase(_STATS_PHRASINGS, stat=_rng.choice(_STAT_NAMES[stat]), numbers=numbers)),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="statistics", args={"numbers": numbers, "stat": stat}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="date_diff",
            task=Task(task_id=f"datediff_{i:04d}", description=_phrase(_DATE_DIFF_PHRASINGS, d1=d1.isoformat(), d2=d2.isoformat())),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="date_diff", args={"date1": d1.isoformat(), "date2": d2.isoformat()}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

    return _unique(n, make)


_EVENT_TITLES = [
    "Doctor appointment", "Project review", "Client call", "Gym session", "Book club", "Dentist visit",
    "Team retrospective", "Lunch with Priya", "Budget planning", "Yoga class", "Design sync", "Parent-teacher meeting",
    "Car service", "Interview panel", "Coffee with Oscar", "Quarterly review", "Piano lesson", "Haircut",
    "Vendor demo", "Birthday dinner",
]


_CALENDAR_CREATE_PHRASINGS = [
    "Schedule an event called '{title}' at {time}, then confirm it's on the calendar.",
    "Add '{title}' to my calendar for {time} and then check that it shows up.",
    "Put '{title}' on my calendar at {time}, then list my events to make sure it's there.",
    "Create a calendar event '{title}' at {time} and confirm it was added.",
    "Can you book '{title}' for {time}? Afterwards, verify it's on my calendar.",
    "I need '{title}' scheduled at {time}. Please add it and double-check the calendar.",
]


def calendar_create_tasks(n: int) -> list[LabeledTask]:
    def make(i: int) -> LabeledTask:
        title = _rng.choice(_EVENT_TITLES)
        time_str = f"2024-05-{_rng.randint(2, 31):02d} {_rng.randint(8, 18):02d}:{_rng.choice(['00', '30'])}"
        answer = f"Scheduled '{title}' at {time_str}"
        return LabeledTask(
            template="calendar_create",
            task=Task(
                task_id=f"cal_create_{i:04d}",
                description=_phrase(_CALENDAR_CREATE_PHRASINGS, title=title, time=time_str),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="calendar_create", args={"time": time_str, "title": title}),
                Action(tool_name="calendar_list", args={}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
    def make(i: int) -> LabeledTask:
        content = _rng.choice(_NOTE_CONTENTS)
        path = f"{_rng.choice(_FILE_STEMS)}.txt"
        return LabeledTask(
            template="file_write_read",
            task=Task(
                task_id=f"file_wr_{i:04d}",
                description=_phrase(_FILE_WRITE_READ_PHRASINGS, content=content, path=path),
            ),
            expected_answer=content,
            expected_actions=[
                Action(tool_name="file_write", args={"path": path, "content": content}),
                Action(tool_name="file_read", args={"path": path}),
                Action(tool_name="finish", args={"answer": content}),
            ],
        )

    return _unique(n, make)


_EMAIL_SUBJECTS = [
    "Meeting follow-up", "Project update", "Quick question", "Invoice attached", "Schedule change",
    "Welcome aboard", "Action items", "Weekly summary", "Contract review", "Travel plans",
]
_EMAIL_BODIES = [
    "Please let me know if you have any questions.",
    "Thanks for your help on this.",
    "Can we find time to discuss this week?",
    "The details are below, let me know what you think.",
    "Just a quick reminder about our deadline.",
    "Looking forward to hearing from you.",
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
        to = _random_email_address()
        subject = _rng.choice(_EMAIL_SUBJECTS)
        body = _rng.choice(_EMAIL_BODIES)
        answer = f"Email sent to {to}"
        return LabeledTask(
            template="email",
            task=Task(
                task_id=f"email_{i:04d}",
                description=_phrase(_EMAIL_PHRASINGS, to=to, subject=subject, body=body),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="send_email", args={"to": to, "subject": subject, "body": body}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="convert_then_prime",
            task=Task(
                task_id=f"conv_prime_{i:04d}",
                description=_phrase(_CONVERT_THEN_PRIME_PHRASINGS, value=value, from_unit=from_unit, to_unit=to_unit),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="converter", args={"value": value, "from_unit": from_unit, "to_unit": to_unit}),
                Action(tool_name="prime_checker", args={"number": converted}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="convert_sum_prime",
            task=Task(
                task_id=f"conv_sum_prime_{i:04d}",
                description=_phrase(
                    _CONVERT_SUM_PRIME_PHRASINGS, v1=v1, from1=from1, to1=to1, v2=v2, from2=from2, to2=to2
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
        )

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
        mean_val = round(stats_module.mean(numbers))
        mean_val = max(mean_val, 1)
        is_prime = "yes" if checker.execute(mean_val).is_prime else "no"
        roman = converter.execute(mean_val).roman
        answer = f"mean rounds to {mean_val}, which is {'' if is_prime == 'yes' else 'not '}prime, and is written {roman} in Roman numerals"
        return LabeledTask(
            template="stats_prime_roman",
            task=Task(
                task_id=f"stats_prime_roman_{i:04d}",
                description=_phrase(_STATS_PRIME_ROMAN_PHRASINGS, numbers=numbers),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="statistics", args={"numbers": numbers, "stat": "mean"}),
                Action(tool_name="prime_checker", args={"number": mean_val}),
                Action(tool_name="roman_numeral", args={"value": mean_val}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        price = _PRICES[ticker]
        to = _random_email_address()
        body = f"Weather in {w.city}: {w.temperature_f}F, {w.conditions}. {ticker.upper()} is trading at ${price}."
        answer = f"Email sent to {to}"
        return LabeledTask(
            template="weather_stock_email",
            task=Task(
                task_id=f"weather_stock_email_{i:04d}",
                description=_phrase(_WEATHER_STOCK_EMAIL_PHRASINGS, city=w.city, ticker=ticker.upper(), to=to),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="weather", args={"city": w.city}),
                Action(tool_name="stock_price", args={"ticker": ticker}),
                Action(tool_name="send_email", args={"to": to, "subject": "Daily summary", "body": body}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="weather_compare",
            task=Task(task_id=f"weather_cmp_{i:04d}", description=_phrase(_WEATHER_COMPARE_PHRASINGS, city1=d1.city, city2=d2.city)),
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
        )

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
        return LabeledTask(
            template="temperature_converter",
            task=Task(task_id=f"temp_{i:04d}", description=_phrase(_TEMPERATURE_PHRASINGS, value=value, from_unit=from_unit, to_unit=to_unit)),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="temperature_converter", args={"value": value, "from_unit": from_unit, "to_unit": to_unit}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="base64_encode",
            task=Task(task_id=f"b64enc_{i:04d}", description=_phrase(_BASE64_ENCODE_PHRASINGS, text=text)),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="base64_encode", args={"text": text}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="base64_decode",
            task=Task(task_id=f"b64dec_{i:04d}", description=_phrase(_BASE64_DECODE_PHRASINGS, encoded=encoded)),
            expected_answer=original,
            expected_actions=[
                Action(tool_name="base64_decode", args={"text": encoded}),
                Action(tool_name="finish", args={"answer": original}),
            ],
        )

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
        return LabeledTask(
            template="encode_decode_roundtrip",
            task=Task(
                task_id=f"b64_roundtrip_{i:04d}",
                description=_phrase(_ROUNDTRIP_PHRASINGS, text=text),
            ),
            expected_answer=text,
            expected_actions=[
                Action(tool_name="base64_encode", args={"text": text}),
                Action(tool_name="base64_decode", args={"text": encoded}),
                Action(tool_name="finish", args={"answer": text}),
            ],
        )

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
    pairs = list(_DISTANCES.keys())
    def make(i: int) -> LabeledTask:
        pair = _rng.choice(pairs)
        # Sort first: a frozenset's iteration order depends on string hashing, which changes every run.
        city1, city2 = _rng.sample(sorted(pair), 2)
        distance = _DISTANCES[pair]
        answer = f"{distance} miles"
        return LabeledTask(
            template="distance_between",
            task=Task(task_id=f"dist_{i:04d}", description=_phrase(_DISTANCE_PHRASINGS, city1=city1.title(), city2=city2.title())),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="distance_between", args={"city1": city1, "city2": city2}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        final = tool.execute(principal, rate, years).final_amount
        answer = str(final)
        return LabeledTask(
            template="compound_interest",
            task=Task(
                task_id=f"compound_{i:04d}",
                description=_phrase(
                    _COMPOUND_INTEREST_PHRASINGS, principal=principal, rate=rate, duration=_plural(years, "year")
                ),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="compound_interest", args={"principal": principal, "rate_pct": rate, "years": years}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="filter_records",
            task=Task(
                task_id=f"filter_{i:04d}",
                description=_phrase(_FILTER_RECORDS_PHRASINGS, records=records, threshold=threshold),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="filter_records", args={"records": records, "threshold": threshold}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="weather_forecast",
            task=Task(task_id=f"forecast_{i:04d}", description=_phrase(_FORECAST_PHRASINGS, days=days, city=city)),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="weather_forecast", args={"city": city, "days": days}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

    return _unique(n, make)


# (expression, function, max argument, ways to say it in words). exp's argument stays small so the
# expression the task asks about, the tool's result, and the expected answer all agree.
_SCI_FUNCTIONS = [
    ("sqrt({n})", math.sqrt, 100, ["the square root of {n}"]),
    ("sin({n})", math.sin, 100, ["the sine of {n} radians", "the sine of {n} (in radians)"]),
    ("cos({n})", math.cos, 100, ["the cosine of {n} radians", "the cosine of {n} (in radians)"]),
    ("log({n})", math.log, 100, ["the natural logarithm of {n}", "the natural log of {n}"]),
    ("exp({n})", math.exp, 10, ["e raised to the power of {n}", "e to the {n}"]),
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
    def make(i: int) -> LabeledTask:
        expr_template, fn, max_n, word_forms = _rng.choice(_SCI_FUNCTIONS)
        n_val = _rng.randint(1, max_n)
        expression = expr_template.format(n=n_val)
        answer = str(round(fn(n_val), 4))
        if _rng.random() < 0.5:
            description = _phrase(_SCI_PHRASINGS, e=expression)
        else:
            description = _phrase(_SCI_PHRASINGS, e=_rng.choice(word_forms).format(n=n_val))
        return LabeledTask(
            template="scientific_calculator",
            task=Task(task_id=f"sci_calc_{i:04d}", description=description),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="scientific_calculator", args={"expression": expression}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="percentile",
            task=Task(task_id=f"pctl_{i:04d}", description=_phrase(_PERCENTILE_PHRASINGS, p=p, numbers=numbers)),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="percentile", args={"numbers": numbers, "p": p}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        groups = tool.execute(records).groups
        answer = "; ".join(f"{g.group}: {g.total}" for g in groups)
        return LabeledTask(
            template="aggregate_records",
            task=Task(
                task_id=f"agg_{i:04d}",
                description=_phrase(_AGGREGATE_PHRASINGS, records=records),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="aggregate_records", args={"records": records}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

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
        return LabeledTask(
            template="add_days",
            task=Task(
                task_id=f"add_days_{i:04d}",
                description=_phrase(
                    _ADD_DAYS_AFTER_PHRASINGS if days >= 0 else _ADD_DAYS_BEFORE_PHRASINGS,
                    days=_plural(abs(days), "day"),
                    start=start.isoformat(),
                ),
            ),
            expected_answer=answer,
            expected_actions=[
                Action(tool_name="add_days", args={"start_date": start.isoformat(), "days": days}),
                Action(tool_name="finish", args={"answer": answer}),
            ],
        )

    return _unique(n, make)


def _weather_lookup():
    d = _rng.choice(list(_WEATHER_DATA.values()))
    return "weather", {"city": d.city}, _phrase(["the weather in {c}", "the current conditions in {c}", "what the weather is like in {c}"], c=d.city), f"weather in {d.city} is {d.temperature_f}F and {d.conditions}"


def _stock_lookup():
    ticker = _rng.choice(list(_PRICES.keys()))
    return "stock_price", {"ticker": ticker}, _phrase(["the price of {t} stock", "what {t} is trading at", "the current {t} share price"], t=ticker.upper()), f"{ticker.upper()} stock is ${_PRICES[ticker]}"


def _prime_lookup():
    number = _rng.randint(1000, 99999)
    is_prime = PrimeCheckerTool().execute(number).is_prime
    return "prime_checker", {"number": number}, _phrase(["whether {n} is prime", "if {n} is a prime number"], n=number), f"{number} is {'prime' if is_prime else 'not prime'}"


def _roman_lookup():
    value = _rng.randint(50, 3999)
    roman = RomanNumeralTool().execute(value).roman
    return "roman_numeral", {"value": value}, _phrase(["the Roman numeral for {v}", "how {v} is written in Roman numerals"], v=value), f"{value} in Roman numerals is {roman}"


def _word_count_lookup():
    text = " ".join(_rng.sample(_SENTENCE_POOL, _rng.randint(2, 3)))
    return "word_count", {"text": text}, _phrase(['how many words are in "{t}"', 'the word count of "{t}"'], t=text), f"the text has {len(text.split())} words"


def _date_diff_lookup():
    base = date(2024, 1, 1)
    d1 = base + timedelta(days=_rng.randint(0, 700))
    d2 = d1 + timedelta(days=_rng.randint(1, 400))
    days = (d2 - d1).days
    question = _phrase(
        ["how many days are between {d1} and {d2}", "the number of days from {d1} to {d2}"],
        d1=d1.isoformat(), d2=d2.isoformat(),
    )
    return "date_diff", {"date1": d1.isoformat(), "date2": d2.isoformat()}, question, f"there are {days} days between those dates"


def _converter_lookup():
    pairs = list(_CONVERSION_RATES.keys())
    from_unit, to_unit = _rng.choice(pairs)
    value = round(_rng.uniform(1, 500), 2)
    converted = round(value * _rate(from_unit, to_unit), 2)
    question = _phrase(
        ["what {v} {f} is in {t}", "how many {t} are in {v} {f}", "{v} {f} expressed in {t}"],
        v=value, f=from_unit, t=to_unit,
    )
    return "converter", {"value": value, "from_unit": from_unit, "to_unit": to_unit}, question, f"{value} {from_unit} is {converted} {to_unit}"


def _temperature_lookup():
    units = ["celsius", "fahrenheit", "kelvin"]
    from_unit, to_unit = _rng.sample(units, 2)
    value = round(_rng.uniform(-40, 100), 1)
    converted = round(convert_temperature(value, from_unit, to_unit), 2)
    question = _phrase(
        ["what {v} degrees {f} is in {t}", "{v} degrees {f} converted to {t}"], v=value, f=from_unit, t=to_unit
    )
    return "temperature_converter", {"value": value, "from_unit": from_unit, "to_unit": to_unit}, question, f"{value} {from_unit} is {converted} {to_unit}"


def _stats_lookup():
    numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
    stat = _rng.choice(["mean", "median", "stdev"])
    value = round(getattr(stats_module, stat)(numbers), 2)
    question = f"the {_rng.choice(_STAT_NAMES[stat])} of {numbers}"
    return "statistics", {"numbers": numbers, "stat": stat}, question, f"the {stat} of that list is {value}"


def _base64_encode_lookup():
    text = _random_text()
    encoded = Base64EncodeTool().execute(text).encoded
    question = _phrase(['the base64 encoding of "{t}"', 'what "{t}" looks like in base64'], t=text)
    return "base64_encode", {"text": text}, question, f'"{text}" encodes to {encoded}'


def _distance_lookup():
    pair = _rng.choice(list(_DISTANCES.keys()))
    city1, city2 = _rng.sample(sorted(pair), 2)  # sorted: frozenset order changes every run
    distance = _DISTANCES[pair]
    question = _phrase(
        ["the distance between {c1} and {c2}", "how far {c1} is from {c2}"], c1=city1.title(), c2=city2.title()
    )
    return "distance_between", {"city1": city1, "city2": city2}, question, f"the distance is {distance} miles"


def _compound_interest_lookup():
    tool = CompoundInterestTool()
    principal = round(_rng.uniform(100, 10000), 2)
    rate = round(_rng.uniform(1, 10), 1)
    years = _rng.randint(1, 20)
    final = tool.execute(principal, rate, years).final_amount
    question = _phrase(
        [
            "the final amount after investing ${p} at {r}% for {d} compounded annually",
            "what ${p} grows to in {d} at {r}% compounded yearly",
        ],
        p=principal, r=rate, d=_plural(years, "year"),
    )
    return "compound_interest", {"principal": principal, "rate_pct": rate, "years": years}, question, f"the final amount is {final}"


def _percentile_lookup():
    numbers = [round(_rng.uniform(1, 100), 1) for _ in range(_rng.randint(5, 12))]
    p = _rng.choice([10, 25, 50, 75, 90, 95])
    value = round(compute_percentile(numbers, p), 2)
    question = _phrase(["the {p}th percentile of {nums}", "the p{p} of {nums}"], p=p, nums=numbers)
    return "percentile", {"numbers": numbers, "p": p}, question, f"the {p}th percentile is {value}"


def _add_days_lookup():
    base = date(2024, 1, 1)
    start = base + timedelta(days=_rng.randint(0, 700))
    days = _rng.randint(-90, 90)
    result = AddDaysTool().execute(start.isoformat(), days).result_date
    question = f"the date {_plural(abs(days), 'day')} {'after' if days >= 0 else 'before'} {start.isoformat()}"
    return "add_days", {"start_date": start.isoformat(), "days": days}, question, f"the resulting date is {result}"


def _scientific_calculator_lookup():
    expr_template, fn, max_n, word_forms = _rng.choice(_SCI_FUNCTIONS)
    n = _rng.randint(1, max_n)
    expression = expr_template.format(n=n)
    value = round(fn(n), 4)
    question = _rng.choice([f"what {expression} evaluates to", word_forms[0].format(n=n)])
    return "scientific_calculator", {"expression": expression}, question, f"{expression} = {value}"


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
        results = [gen() for gen in _rng.sample(_LOOKUP_GENERATORS, k)]
        actions = [Action(tool_name=name, args=args) for name, args, _, _ in results]
        description = _phrase(_MIX_PHRASINGS, questions="; and ".join(q for _, _, q, _ in results))
        answer = "; ".join(a for _, _, _, a in results)
        return LabeledTask(
            template=f"mix_k{k}",
            task=Task(task_id=f"mix{k}_{i:04d}", description=description),
            expected_answer=answer,
            expected_actions=[*actions, Action(tool_name="finish", args={"answer": answer})],
        )

    return _unique(n, make)


_BASE_COUNTS: dict[str, int] = {
    "calculator": 200, "converter": 150, "weather": 100, "stock_price": 100,
    "word_count": 120, "prime_checker": 120, "roman_numeral": 100,
    "stats": 120, "date_diff": 100,
    "temperature_converter": 120, "base64_encode": 100, "base64_decode": 100,
    "distance_between": 100, "compound_interest": 120, "filter_records": 100,
    "weather_forecast": 100, "scientific_calculator": 120, "percentile": 100,
    "aggregate_records": 100, "add_days": 100,
    "calendar_create": 150, "file_write_read": 150, "email": 150, "convert_then_prime": 150,
    "weather_compare": 100, "encode_decode_roundtrip": 150, "convert_sum_prime": 150,
    "stats_prime_roman": 150, "weather_stock_email": 150,
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
        # dependency chains (2-4 tools, each step needs the previous step's result)
        *calendar_create_tasks(n["calendar_create"]),
        *file_write_read_tasks(n["file_write_read"]),
        *email_tasks(n["email"]),
        *convert_then_prime_tasks(n["convert_then_prime"]),
        *weather_compare_tasks(n["weather_compare"]),
        *encode_decode_roundtrip_tasks(n["encode_decode_roundtrip"]),
        *convert_sum_prime_tasks(n["convert_sum_prime"]),
        *stats_prime_roman_tasks(n["stats_prime_roman"]),
        *weather_stock_email_tasks(n["weather_stock_email"]),
        # independent-lookup mixes (no dependency between the tool calls)
        *mixed_lookup_tasks(n["mix_k2"], k=2),
        *mixed_lookup_tasks(n["mix_k3"], k=3),
        *mixed_lookup_tasks(n["mix_k4"], k=4),
        *mixed_lookup_tasks(n["mix_k5"], k=5),
    ]
