from app.tools.base import Tool, ToolMetadata
from typing import Any
from pydantic import BaseModel, Field
import time


class ToolExecutionResult(BaseModel):
    metadata: ToolMetadata = Field(description="Metadata of the tool that was run.")
    args: dict[str, Any] = Field(description="Keyword arguments the tool was called with.")
    latency: float = Field(description="Wall-clock time the call took, in seconds.")
    success: bool = Field(description="Whether the tool's execute() completed without raising.")
    cost: float | None = Field(description="Cost of the call, as reported by the tool's cost() method, or None if it failed.")
    error: str | None = Field(description="Error message if execute() raised, or None on success.")
    output: dict[str, Any] | None = Field(description="The tool's return value on success (as a plain dict), or None if it failed.")


class ToolExecutor:
    def execute(self, tool: Tool, **kwargs: Any):
        start = time.perf_counter()
        try:
            result = tool.execute(**kwargs)
            output = result.model_dump()
            cost = tool.cost(**kwargs)
            success = True
            error = None
        except Exception as e:
            output = None
            cost = None
            success = False
            error = str(e)
        latency = time.perf_counter() - start

        return ToolExecutionResult(
            metadata=tool.metadata,
            args=kwargs,
            latency=latency,
            success=success,
            cost=cost,
            error=error,
            output=output
        )