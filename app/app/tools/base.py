from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Literal, Any


type ParamType = Literal[
    "string",
    "integer",
    "number",
    "boolean",
    "array",
    "object",
]


class ToolParameter(BaseModel):
    name: str = Field(description="Parameter name, as it appears in the arguments dict passed to execute().")
    type: ParamType = Field(description="JSON-schema-style type of the parameter's value.")
    description: str = Field(description="Human/LLM-readable explanation of what this parameter means and how to set it.")


class ToolMetadata(BaseModel):
    name: str = Field(description="Unique identifier for the tool, used for lookup in the registry and for tool selection.")
    description: str = Field(description="Human/LLM-readable explanation of what the tool does and when to use it.")
    parameters: list[ToolParameter] = Field(description="Ordered list of parameters the tool's execute() accepts.")
    returns: str = Field(description="Name of the Pydantic model class describing the shape of execute()'s return value.")


class Tool(ABC):
    metadata: ToolMetadata

    @abstractmethod
    def execute(self, **kwargs: Any) -> BaseModel:
        ...

    @abstractmethod
    def cost(self, **kwargs: Any) -> float:
        ...

    def reset(self) -> None:
        """Override for stateful tools to reset their state at the start of an episode. No-op by default."""
        pass