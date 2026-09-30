import random
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "user", "assistant"]


class Provider(StrEnum):
    GROQ = "groq"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatMessage(_StrictModel):
    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class GenerationConfig(_StrictModel):
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, gt=0)
    top_p: float = Field(default=1.0, ge=0, le=1)
    timeout_s: float = Field(default=30.0, gt=0)


class RetryPolicy(_StrictModel):
    max_attempts: int = Field(default=3, ge=1)
    base_delay_s: float = Field(default=1.0, ge=0)
    max_delay_s: float = Field(default=20.0, ge=0)
    jitter: bool = True

    def delay_for(self, attempt: int, retry_after_s: float | None = None) -> float:
        if retry_after_s is not None:
            return min(retry_after_s, self.max_delay_s)
        delay = self.base_delay_s * 2.0 ** (attempt - 1)
        if self.jitter:
            delay += random.uniform(0, delay * 0.1)
        return min(delay, self.max_delay_s)


class Usage(_StrictModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


ErrorKind = Literal[
    "auth", "rate_limit", "timeout", "network", "server", "invalid_request", "unknown"
]


class LLMError(_StrictModel):
    kind: ErrorKind
    message: str
    provider: Provider
    retryable: bool
    status_code: int | None = None
    retry_after_s: float | None = None


class ModelResponse(_StrictModel):
    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
