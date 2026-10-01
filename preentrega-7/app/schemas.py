from datetime import UTC, datetime
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


type TraceStep = Annotated[Delegation | Contribution, Field(discriminator="kind")]


class Segment(BaseModel):
    path: list[str] = []
    steps: list[TraceStep] = []
    final_answer: str | None = None
    finish_reason: str | None = None
    draft: str | None = None
    published: bool | None = None


class Approval(BaseModel):
    approved: bool
    approver: str = Field(min_length=1)
    comment: str = ""


type JobStatus = Literal["PENDING", "RUNNING", "PAUSED_FOR_APPROVAL", "DONE", "FAILED"]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Job(BaseModel):
    id: str
    status: JobStatus
    question: str
    created_at: datetime
    started_at: datetime | None = None
    paused_at: datetime | None = None
    decided_at: datetime | None = None
    finished_at: datetime | None = None
    run_seconds: float = 0.0
    steps: list[TraceStep] = []
    draft: str | None = None
    final_answer: str | None = None
    published: bool | None = None
    approval: Approval | None = None
    error: str | None = None
    usage: Usage = Usage()

    def fail(self, error: str) -> None:
        self.status = "FAILED"
        self.error = error
        self.finished_at = datetime.now(UTC)
