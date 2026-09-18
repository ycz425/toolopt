from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


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
        parameters=[],
        returns=ListEventsResult.__name__,
    )

    def __init__(self, store: CalendarStore):
        self.store = store

    def execute(self) -> ListEventsResult:
        return ListEventsResult(events=list(self.store.events))

    def cost(self) -> float:
        return 0.01

    def reset(self) -> None:
        self.store.reset()


class CreateEventResult(BaseModel):
    event: CalendarEvent = Field(description="The event that was created.")


class CalendarCreateTool(Tool):
    metadata = ToolMetadata(
        name="calendar_create",
        description="Create a new event on the calendar.",
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

    def cost(self, time: str, title: str) -> float:
        return 0.02

    def reset(self) -> None:
        self.store.reset()
