from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter, ToolPricing


class CalendarEvent(BaseModel):
    time: str = Field(description="Event time, e.g. '2024-05-01 14:00'.")
    title: str = Field(description="Short title of the event.")


class CalendarStore:
    def __init__(self):
        self.events: list[CalendarEvent] = []
        self.reset()

    def reset(self) -> None:
        self.events = [
            CalendarEvent(time="2024-05-01 09:00", title="Team standup"),
            CalendarEvent(time="2024-05-01 13:00", title="Lunch with Sam"),
        ]


class ListEventsResult(BaseModel):
    events: list[CalendarEvent] = Field(description="All events currently on the calendar.")


class CalendarListTool(Tool):
    metadata = ToolMetadata(
        name="calendar_list",
        description="List all events currently on the calendar.",
        pricing=ToolPricing(base=0.01),
        parameters=[],
        returns=ListEventsResult.__name__,
    )

    def __init__(self, store: CalendarStore):
        self.store = store

    def execute(self) -> ListEventsResult:
        return ListEventsResult(events=list(self.store.events))

    def reset(self) -> None:
        self.store.reset()


class CreateEventResult(BaseModel):
    event: CalendarEvent = Field(description="The event that was created.")


class CalendarCreateTool(Tool):
    metadata = ToolMetadata(
        name="calendar_create",
        description="Create a new event on the calendar.",
        pricing=ToolPricing(base=0.02),
        parameters=[
            ToolParameter(name="time", type="string", description="Event time, e.g. '2024-05-01 14:00'."),
            ToolParameter(name="title", type="string", description="Short title of the event."),
        ],
        returns=CreateEventResult.__name__,
    )

    def __init__(self, store: CalendarStore):
        self.store = store

    def execute(self, time: str, title: str) -> CreateEventResult:
        event = CalendarEvent(time=time, title=title)
        self.store.events.append(event)
        return CreateEventResult(event=event)

    def reset(self) -> None:
        self.store.reset()


class CreateEventVerifiedResult(BaseModel):
    event: CalendarEvent = Field(description="The event that was created.")
    events: list[CalendarEvent] = Field(description="All events on the calendar after the event was created.")


class CalendarCreateVerifiedTool(Tool):
    metadata = ToolMetadata(
        name="calendar_create_verified",
        description="Create a new event on the calendar, then return the full calendar so you can confirm it was added.",
        pricing=ToolPricing(base=0.025),
        parameters=[
            ToolParameter(name="time", type="string", description="Event time, e.g. '2024-05-01 14:00'."),
            ToolParameter(name="title", type="string", description="Short title of the event."),
        ],
        returns=CreateEventVerifiedResult.__name__,
    )

    def __init__(self, store: CalendarStore):
        self.store = store

    def execute(self, time: str, title: str) -> CreateEventVerifiedResult:
        event = CalendarEvent(time=time, title=title)
        self.store.events.append(event)
        return CreateEventVerifiedResult(event=event, events=list(self.store.events))

    def reset(self) -> None:
        self.store.reset()
