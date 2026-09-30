from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import groq
import httpx

from llm_client.groq_client import GroqClient
from llm_client.schemas import ChatMessage, GenerationConfig, RetryPolicy

_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

HELLO = [ChatMessage(role="user", content="hola")]


def completion(content: str, finish_reason: str = "stop") -> Any:
    return SimpleNamespace(
        model="fake-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def status_error(
    cls: type[groq.APIStatusError], status: int, headers: dict[str, str] | None = None
) -> groq.APIStatusError:
    response = httpx.Response(status, request=_REQUEST, headers=headers)
    return cls("boom", response=response, body=None)


def timeout_error() -> groq.APITimeoutError:
    return groq.APITimeoutError(request=_REQUEST)


def connection_error() -> groq.APIConnectionError:
    return groq.APIConnectionError(request=_REQUEST)


class FakeStream:
    def __init__(self, *pieces: str | None | Exception) -> None:
        self._pieces = pieces
        self.closed = False

    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True

    async def _iterate(self) -> AsyncIterator[Any]:
        for piece in self._pieces:
            if isinstance(piece, Exception):
                raise piece
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()


class FakeGroq:
    """Stands in for AsyncGroq: each call to create() consumes the next scripted outcome."""

    def __init__(self, *outcomes: Any) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(sdk: FakeGroq) -> GroqClient:
    return GroqClient(
        GenerationConfig(model="fake-model"),
        retry_policy=RetryPolicy(base_delay_s=0, jitter=False),
        sdk=sdk,  # type: ignore[arg-type]
    )
