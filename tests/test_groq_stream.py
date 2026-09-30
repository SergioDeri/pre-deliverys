from collections.abc import AsyncIterator

import groq
import httpx
import pytest

from llm_client.errors import LLMStreamError

from .fakes import HELLO, FakeGroq, FakeStream, connection_error, make_client, status_error


async def collect(stream: AsyncIterator[str]) -> list[str]:
    return [fragment async for fragment in stream]


async def test_stream_yields_text_fragments_in_order() -> None:
    client = make_client(FakeGroq(FakeStream("Ho", None, "la", "", "!")))

    assert await collect(client.stream(HELLO)) == ["Ho", "la", "!"]


async def test_failure_before_the_first_fragment_is_retried() -> None:
    sdk = FakeGroq(
        status_error(groq.RateLimitError, 429),
        FakeStream(connection_error()),
        FakeStream("hola"),
    )
    client = make_client(sdk)

    assert await collect(client.stream(HELLO)) == ["hola"]
    assert len(sdk.calls) == 3


async def test_failure_after_text_was_delivered_is_not_retried() -> None:
    broken = FakeStream("Ho", "la", connection_error())
    sdk = FakeGroq(broken, FakeStream("never reached"))
    client = make_client(sdk)
    received: list[str] = []

    with pytest.raises(LLMStreamError) as caught:
        async for fragment in client.stream(HELLO):
            received.append(fragment)

    assert received == ["Ho", "la"]
    assert caught.value.error.kind == "network"
    assert len(sdk.calls) == 1
    assert broken.closed


async def test_invalid_api_key_ends_the_stream_with_a_structured_error() -> None:
    sdk = FakeGroq(status_error(groq.AuthenticationError, 401))
    client = make_client(sdk)

    with pytest.raises(LLMStreamError) as caught:
        await collect(client.stream(HELLO))

    assert caught.value.error.kind == "auth"
    assert caught.value.error.retryable is False


async def test_transport_error_while_reading_is_retried_before_the_first_fragment() -> None:
    sdk = FakeGroq(FakeStream(httpx.RemoteProtocolError("peer closed")), FakeStream("hola"))
    client = make_client(sdk)

    assert await collect(client.stream(HELLO)) == ["hola"]


async def test_read_timeout_after_text_ends_the_stream_with_a_structured_error() -> None:
    client = make_client(FakeGroq(FakeStream("Ho", httpx.ReadTimeout("slow"))))

    with pytest.raises(LLMStreamError) as caught:
        await collect(client.stream(HELLO))

    assert caught.value.error.kind == "timeout"


def test_empty_conversation_is_rejected_as_soon_as_the_stream_is_requested() -> None:
    client = make_client(FakeGroq())

    with pytest.raises(ValueError):
        client.stream([])
