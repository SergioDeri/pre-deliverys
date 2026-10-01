from typing import Any

from pydantic import BaseModel


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    name: str
    content: str


class Step(BaseModel):
    node: str
    reasoning: str | None = None
    tool_calls: list[ToolCallRecord] = []
    tool_results: list[ToolResult] = []
    content: str | None = None


class TurnTrace(BaseModel):
    thread_id: str
    question: str
    steps: list[Step] = []
    answer: str | None = None
    error: str | None = None
