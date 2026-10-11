from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Any


ParamType = Literal[
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


class ToolPricing(BaseModel):
    base: float = Field(ge=0, description="Charged once per call.")
    per_unit: float = Field(default=0.0, ge=0, description="Charged per unit of unit_param; 0 for flat pricing.")
    unit_param: str | None = Field(default=None, description="Name of the parameter that determines the unit count.")


class ToolMetadata(BaseModel):
    name: str = Field(description="Unique identifier for the tool, used for lookup in the registry and for tool selection.")
    description: str = Field(description="Human/LLM-readable explanation of what the tool does and when to use it.")
    pricing: ToolPricing = Field(description="Documented price of a call, shown to the model. The actual charge comes from Tool.cost().")
    parameters: list[ToolParameter] = Field(description="Ordered list of parameters the tool's execute() accepts.")
    returns: str = Field(description="Name of the Pydantic model class describing the shape of execute()'s return value.")

    @model_validator(mode="after")
    def _check_pricing(self) -> "ToolMetadata":
        """Rejects pricing that Tool.cost() can't charge, so a mispriced tool fails when it's defined, not mid-run."""
        unit_param = self.pricing.unit_param
        if unit_param is None:
            if self.pricing.per_unit > 0:
                raise ValueError(f"tool '{self.name}': per_unit is set but unit_param isn't, so per-unit charges would never apply")
            return self
        param = next((p for p in self.parameters if p.name == unit_param), None)
        if param is None:
            raise ValueError(f"tool '{self.name}': pricing.unit_param '{unit_param}' is not one of its parameters "
                             f"({[p.name for p in self.parameters]})")
        if param.type not in ("integer", "number", "string", "array"):
            raise ValueError(f"tool '{self.name}': pricing.unit_param '{unit_param}' has type '{param.type}', "
                             "which can't be counted; use an integer, number, string, or array parameter")
        return self


class Tool(ABC):
    metadata: ToolMetadata

    @abstractmethod
    def execute(self, **kwargs: Any) -> BaseModel:
        ...

    def cost(self, **kwargs: Any) -> float:
        pricing = self.metadata.pricing

        if pricing.unit_param is None or pricing.unit_param not in kwargs:
            return pricing.base

        param = [p for p in self.metadata.parameters if p.name == pricing.unit_param][0]
        if param.type in {'integer', 'number'}:
            f = lambda x: abs(x)
        elif param.type in {'string', 'array'}:
            f = lambda x: len(x)

        # The check in ToolMetadata ensures pricing.unit_param is always a valid parameter name
        # and is either an integer, number, string, or array

        return pricing.base + f(kwargs[pricing.unit_param]) * pricing.per_unit
        

    def reset(self) -> None:
        """Override for stateful tools to reset their state at the start of an episode. No-op by default."""
        pass