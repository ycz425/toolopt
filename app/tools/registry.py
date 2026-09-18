from __future__ import annotations

from app.tools.base import Tool
from app.tools.builtin.finish import FinishTool


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self.register(FinishTool())

    def register(self, tool: Tool):
        if tool.metadata.name in self.tools:
            raise ValueError(f'tool {tool.metadata.name} already exists')
        self.tools[tool.metadata.name] = tool

    def remove(self, name: str):
        if name not in self.tools:
            raise ValueError(f'tool {name} not found')
        del self.tools[name]

    def get(self, name: str):
        if name not in self.tools:
            raise ValueError(f'tool {name} not found')
        return self.tools[name]

    def list(self):
        return [self.tools[tool].metadata for tool in self.tools]

    def all_tools(self) -> list[Tool]:
        return list(self.tools.values())

        