from typing import Any

import groq

import pytest

from llm_client.schemas import GenerationConfig

from .fakes import (
    HELLO,
    FakeGroq,
    completion,
    connection_error,
    make_client,
    status_error,
    timeout_error,
)


async def test_generate_returns_the_model_answer_with_usage() -> None:
    client = make_client(FakeGroq(completion("¡Hola!")))

    response = await client.generate(HELLO)

    assert response.error is None
    assert response.content == "¡Hola!"
    assert response.provider == "groq"
    assert response.finish_reason == "stop"
    assert response.usage is not None and response.usage.total_tokens == 8
    assert response.attempts == 1


async def test_invalid_api_key_is_reported_without_retrying() -> None:
    sdk = FakeGroq(status_error(groq.AuthenticationError, 401), completion("never reached"))
    client = make_client(sdk)

    response = await client.generate(HELLO)

    assert not response.ok
    assert response.error is not None
    assert response.error.kind == "auth"
    assert response.error.retryable is False
    assert response.error.status_code == 401
    assert response.attempts == 1
    assert len(sdk.calls) == 1


@pytest.mark.parametrize(
    ("failure", "kind"),
    [
        (lambda: status_error(groq.RateLimitError, 429), "rate_limit"),
        (lambda: status_error(groq.InternalServerError, 503), "server"),
        (timeout_error, "timeout"),
        (connection_error, "network"),
    ],
)
async def test_retryable_errors_are_retried_until_success(failure: Any, kind: str) -> None:
    sdk = FakeGroq(failure(), completion("por fin"))
    client = make_client(sdk)

    response = await client.generate(HELLO)

    assert response.ok
    assert response.content == "por fin"
    assert response.attempts == 2


async def test_gives_up_after_the_last_attempt_and_reports_the_error() -> None:
    sdk = FakeGroq(*(status_error(groq.RateLimitError, 429) for _ in range(3)))
    client = make_client(sdk)

    response = await client.generate(HELLO)

    assert response.error is not None
    assert response.error.kind == "rate_limit"
    assert response.attempts == 3
    assert len(sdk.calls) == 3


async def test_bad_request_is_not_retried() -> None:
    sdk = FakeGroq(status_error(groq.BadRequestError, 400), completion("never reached"))
    client = make_client(sdk)

    response = await client.generate(HELLO)

    assert response.error is not None
    assert response.error.kind == "invalid_request"
    assert len(sdk.calls) == 1


async def test_rate_limit_error_carries_the_server_retry_after() -> None:
    limited = [status_error(groq.RateLimitError, 429, {"retry-after": "0"}) for _ in range(3)]
    client = make_client(FakeGroq(*limited))

    response = await client.generate(HELLO)

    assert response.error is not None
    assert response.error.retry_after_s == 0


async def test_empty_conversation_is_rejected_before_calling_the_api() -> None:
    sdk = FakeGroq()
    client = make_client(sdk)

    with pytest.raises(ValueError):
        await client.generate([])
    assert sdk.calls == []


async def test_per_call_config_overrides_the_client_default() -> None:
    sdk = FakeGroq(completion("ok"))
    client = make_client(sdk)

    await client.generate(HELLO, GenerationConfig(model="other-model", temperature=0))

    assert sdk.calls[0]["model"] == "other-model"
    assert sdk.calls[0]["temperature"] == 0
