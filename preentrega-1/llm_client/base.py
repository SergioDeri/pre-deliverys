import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import aclosing
from types import TracebackType
from typing import ClassVar, Self

from .errors import LLMStreamError
from .schemas import (
    ChatMessage,
    GenerationConfig,
    LLMError,
    ModelResponse,
    Provider,
    RetryPolicy,
)

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    provider: ClassVar[Provider]

    def __init__(self, config: GenerationConfig, retry_policy: RetryPolicy | None = None) -> None:
        self.config = config
        self.retry_policy = retry_policy or RetryPolicy()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    @abstractmethod
    async def aclose(self) -> None: ...

    async def generate(
        self, messages: Sequence[ChatMessage], config: GenerationConfig | None = None
    ) -> ModelResponse:
        _require_messages(messages)
        config = config or self.config
        started = time.perf_counter()
        outcome, attempts = await self._with_retries(lambda: self._complete(messages, config))
        if isinstance(outcome, LLMError):
            outcome = ModelResponse(provider=self.provider, model=config.model, error=outcome)
        return outcome.model_copy(
            update={"latency_ms": (time.perf_counter() - started) * 1000, "attempts": attempts}
        )

    def stream(
        self, messages: Sequence[ChatMessage], config: GenerationConfig | None = None
    ) -> AsyncIterator[str]:
        _require_messages(messages)
        return self._stream(messages, config or self.config)

    async def _stream(
        self, messages: Sequence[ChatMessage], config: GenerationConfig
    ) -> AsyncIterator[str]:
        # Only the wait for the first fragment is retried: once text has reached the
        # caller, starting over would repeat it.
        outcome, _ = await self._with_retries(lambda: self._open_with_first(messages, config))
        if isinstance(outcome, LLMError):
            raise LLMStreamError(outcome)
        first, rest = outcome
        async with aclosing(rest):
            if first is None:
                return
            yield first
            try:
                async for fragment in rest:
                    yield fragment
            except Exception as exc:
                raise LLMStreamError(self._translate(exc)) from exc

    async def _open_with_first(
        self, messages: Sequence[ChatMessage], config: GenerationConfig
    ) -> tuple[str | None, AsyncGenerator[str]]:
        fragments = self._stream_fragments(messages, config)
        try:
            return await anext(fragments, None), fragments
        except BaseException:
            await fragments.aclose()
            raise

    async def _with_retries[T](
        self, call: Callable[[], Awaitable[T]]
    ) -> tuple[T | LLMError, int]:
        policy = self.retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            try:
                return await call(), attempt
            except Exception as exc:
                error = self._translate(exc)
                if not error.retryable or attempt == policy.max_attempts:
                    return error, attempt
                delay = policy.delay_for(attempt, error.retry_after_s)
                logger.warning(
                    "%s: %s (intento %d/%d), reintentando en %.1fs",
                    self.provider,
                    error.kind,
                    attempt,
                    policy.max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable: max_attempts >= 1")

    def _translate(self, exc: Exception) -> LLMError:
        error = self._to_llm_error(exc)
        if error is None:
            raise exc
        return error

    @abstractmethod
    async def _complete(
        self, messages: Sequence[ChatMessage], config: GenerationConfig
    ) -> ModelResponse:
        """Single request to the provider; the base class adds retries, latency and attempts."""

    @abstractmethod
    def _stream_fragments(
        self, messages: Sequence[ChatMessage], config: GenerationConfig
    ) -> AsyncGenerator[str]:
        """Yield non-empty text fragments as the provider streams them."""

    @abstractmethod
    def _to_llm_error(self, exc: Exception) -> LLMError | None:
        """Translate a provider exception, or return None to let it propagate as a bug."""


def _require_messages(messages: Sequence[ChatMessage]) -> None:
    if not messages:
        raise ValueError("a conversation needs at least one message")
