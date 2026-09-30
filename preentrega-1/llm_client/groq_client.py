from collections.abc import AsyncGenerator, Sequence
from typing import Any

import groq
import httpx
from groq import AsyncGroq

from .base import BaseLLMClient
from .schemas import (
    ChatMessage,
    ErrorKind,
    GenerationConfig,
    LLMError,
    ModelResponse,
    Provider,
    RetryPolicy,
    Usage,
)


class GroqClient(BaseLLMClient):
    provider = Provider.GROQ

    def __init__(
        self,
        config: GenerationConfig,
        api_key: str | None = None,
        retry_policy: RetryPolicy | None = None,
        sdk: AsyncGroq | None = None,
    ) -> None:
        super().__init__(config, retry_policy)
        # Retries are ours (see RetryPolicy); the SDK's would hide them and double the wait.
        self._sdk = sdk or AsyncGroq(api_key=api_key, max_retries=0)

    async def aclose(self) -> None:
        await self._sdk.close()

    def _request_kwargs(
        self, messages: Sequence[ChatMessage], config: GenerationConfig
    ) -> dict[str, Any]:
        return {
            "model": config.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "timeout": config.timeout_s,
        }

    async def _complete(
        self, messages: Sequence[ChatMessage], config: GenerationConfig
    ) -> ModelResponse:
        completion = await self._sdk.chat.completions.create(
            **self._request_kwargs(messages, config)
        )
        choice = completion.choices[0]
        usage = completion.usage
        return ModelResponse(
            provider=self.provider,
            model=completion.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=Usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
            if usage
            else None,
        )

    async def _stream_fragments(
        self, messages: Sequence[ChatMessage], config: GenerationConfig
    ) -> AsyncGenerator[str]:
        stream = await self._sdk.chat.completions.create(
            **self._request_kwargs(messages, config), stream=True
        )
        async with stream:
            async for chunk in stream:
                if chunk.choices and (text := chunk.choices[0].delta.content):
                    yield text

    def _to_llm_error(self, exc: Exception) -> LLMError | None:
        kind: ErrorKind
        status_code = None
        retry_after_s = None
        match exc:
            # While a stream is being read the SDK lets raw httpx errors through.
            case groq.APITimeoutError() | httpx.TimeoutException():
                kind, retryable = "timeout", True
            case groq.APIConnectionError() | httpx.TransportError():
                kind, retryable = "network", True
            case groq.APIStatusError(status_code=status):
                status_code = status
                kind, retryable = _classify_status(status)
                retry_after_s = _parse_retry_after(exc.response.headers.get("retry-after"))
            case groq.GroqError():
                kind, retryable = "unknown", False
            case _:
                return None
        return LLMError(
            kind=kind,
            message=str(exc),
            provider=self.provider,
            retryable=retryable,
            status_code=status_code,
            retry_after_s=retry_after_s,
        )


def _classify_status(status: int) -> tuple[ErrorKind, bool]:
    if status in (401, 403):
        return "auth", False
    if status == 429:
        return "rate_limit", True
    if status >= 500:
        return "server", True
    if 400 <= status < 500:
        return "invalid_request", False
    return "unknown", False


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
