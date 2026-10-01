from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

type Level = Literal["DEBUG", "INFO", "WARN", "ERROR"]


class LogLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    level: Level
    service: str
    message: str

    def __str__(self) -> str:
        return f"{self.timestamp:%Y-%m-%d %H:%M:%S} {self.level:<5} [{self.service}] {self.message}"


class ErrorCodeEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    codigo: str
    titulo: str
    http: int | None
    causa: str
    runbook: str | None


class Evidence(BaseModel):
    log_lines: list[LogLine] = []
    error_codes: list[ErrorCodeEntry] = []


type Specialist = Literal["researcher", "analyst"]


class Delegation(BaseModel):
    kind: Literal["delegation"] = "delegation"
    agent: Specialist
    instruction: str
    reason: str
    override: bool = False


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]


class Contribution(BaseModel):
    kind: Literal["contribution"] = "contribution"
    agent: Specialist
    content: str
    tool_calls: list[ToolCallRecord] = []


class DelegationTrace(BaseModel):
    question: str
    path: list[str] = []
    steps: list[Annotated[Delegation | Contribution, Field(discriminator="kind")]] = []
    final_answer: str | None = None
    finish_reason: str | None = None
    error: str | None = None
