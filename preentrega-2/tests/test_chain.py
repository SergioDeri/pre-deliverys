import groq
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables.retry import ExponentialJitterParams

from chain import build_chain, extract, extract_many
from schemas import TechnicalAnalysis
from tests.fakes import (
    REQUEST,
    VALID_ARGS,
    ScriptedChatModel,
    status_error,
    tool_reply,
    tool_use_failed,
)

NO_WAIT = ExponentialJitterParams(initial=0, max=0, jitter=0)

LOG = "ERROR api-pedidos: psycopg_pool.PoolTimeout tras 30s; FastAPI devuelve 503."


async def test_valid_reply_becomes_a_technical_analysis() -> None:
    model = ScriptedChatModel(replies=[tool_reply(VALID_ARGS)])

    outcome = await extract(build_chain(model, backoff=NO_WAIT), LOG)

    assert outcome.failure is None
    assert outcome.analysis == TechnicalAnalysis(**VALID_ARGS)
    assert LOG in model.received[0][-1].content


async def test_invalid_reply_is_retried_until_a_valid_one_arrives() -> None:
    invalid = tool_reply(VALID_ARGS | {"tecnologias": []})
    model = ScriptedChatModel(replies=[invalid, tool_reply(VALID_ARGS)])

    outcome = await extract(build_chain(model, backoff=NO_WAIT), LOG)

    assert outcome.analysis == TechnicalAnalysis(**VALID_ARGS)
    assert len(model.received) == 2


async def test_invalid_replies_on_every_attempt_end_in_a_controlled_failure() -> None:
    model = ScriptedChatModel(replies=[tool_reply(VALID_ARGS | {"nivel_de_criticidad": "critica"})])

    outcome = await extract(build_chain(model, backoff=NO_WAIT), "va lento a veces")

    assert outcome.analysis is None
    assert outcome.failure is not None
    assert outcome.failure.kind == "invalid_output"
    assert outcome.failure.message.startswith("nivel_de_criticidad: ")
    assert "\n" not in outcome.failure.message
    assert len(model.received) == 3


async def test_reply_without_tool_call_counts_as_invalid_output() -> None:
    model = ScriptedChatModel(replies=[AIMessage(content="No puedo analizar esto.")])

    outcome = await extract(build_chain(model, max_attempts=1, backoff=NO_WAIT), "hola")

    assert outcome.failure is not None
    assert outcome.failure.kind == "invalid_output"


async def test_truncated_reply_is_retried_and_reported_when_attempts_run_out() -> None:
    truncated = tool_reply(VALID_ARGS | {"resumen_tecnico": "El pool de conexio"}, "length")
    model = ScriptedChatModel(replies=[truncated])

    outcome = await extract(build_chain(model, backoff=NO_WAIT), LOG)

    assert outcome.failure is not None
    assert outcome.failure.kind == "truncated_output"
    assert len(model.received) == 3


async def test_truncated_reply_is_rejected_even_if_it_parses() -> None:
    model = ScriptedChatModel(replies=[tool_reply(VALID_ARGS, "length"), tool_reply(VALID_ARGS)])

    outcome = await extract(build_chain(model, backoff=NO_WAIT), LOG)

    assert outcome.analysis == TechnicalAnalysis(**VALID_ARGS)
    assert len(model.received) == 2


@pytest.mark.parametrize(
    "transient",
    [
        status_error(groq.RateLimitError, 429),
        status_error(groq.InternalServerError, 503),
        groq.APITimeoutError(request=REQUEST),
        groq.APIConnectionError(request=REQUEST),
    ],
)
async def test_transient_provider_errors_are_retried(transient: Exception) -> None:
    model = ScriptedChatModel(replies=[transient, tool_reply(VALID_ARGS)])

    outcome = await extract(build_chain(model, backoff=NO_WAIT), LOG)

    assert outcome.analysis == TechnicalAnalysis(**VALID_ARGS)
    assert len(model.received) == 2


@pytest.mark.parametrize(
    "permanent",
    [
        status_error(groq.AuthenticationError, 401, "Invalid API Key", "invalid_api_key"),
        status_error(groq.BadRequestError, 400, "Please reduce the length of the messages"),
    ],
)
async def test_permanent_provider_errors_fail_on_the_first_attempt(permanent: Exception) -> None:
    model = ScriptedChatModel(replies=[permanent])

    outcome = await extract(build_chain(model, backoff=NO_WAIT), LOG)

    assert outcome.failure is not None
    assert outcome.failure.kind == "provider"
    assert outcome.failure.message.startswith("HTTP ")
    assert "Error code" not in outcome.failure.message
    assert len(model.received) == 1


async def test_failed_tool_call_on_groq_is_retried_as_invalid_output() -> None:
    model = ScriptedChatModel(replies=[tool_use_failed()])

    outcome = await extract(build_chain(model, backoff=NO_WAIT), LOG)

    assert outcome.failure is not None
    assert outcome.failure.kind == "invalid_output"
    assert len(model.received) == 3


async def test_many_texts_are_extracted_independently() -> None:
    model = ScriptedChatModel(replies=[status_error(groq.AuthenticationError, 401)])
    texts = ["primer log", "segunda arquitectura", "tercer texto"]

    outcomes = await extract_many(build_chain(model, backoff=NO_WAIT), texts)

    assert [o.failure.kind if o.failure else None for o in outcomes] == ["provider"] * 3
    assert sorted(m[-1].content for m in model.received) == sorted(
        f"Texto a analizar:\n\n{text}" for text in texts
    )


async def test_retry_after_invalid_reply_tells_the_model_what_was_wrong() -> None:
    invalid = tool_reply(VALID_ARGS | {"tecnologias": []})
    model = ScriptedChatModel(replies=[invalid, tool_reply(VALID_ARGS)])

    await extract(build_chain(model, backoff=NO_WAIT), LOG)

    first, second = model.received
    assert "tecnologias" not in first[-1].content
    assert "tecnologias" in second[-1].content
    assert LOG in second[-2].content


async def test_retry_after_truncated_reply_asks_for_a_shorter_answer() -> None:
    model = ScriptedChatModel(replies=[tool_reply(VALID_ARGS, "length"), tool_reply(VALID_ARGS)])

    await extract(build_chain(model, backoff=NO_WAIT), LOG)

    assert "límite de tokens" in model.received[1][-1].content


async def test_discarded_provider_attempts_are_logged_without_the_raw_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    limited = status_error(groq.RateLimitError, 429, "Rate limit reached", "rate_limit_exceeded")
    model = ScriptedChatModel(replies=[limited, tool_reply(VALID_ARGS)])

    await extract(build_chain(model, backoff=NO_WAIT), LOG)

    assert "HTTP 429: Rate limit reached" in caplog.text
    assert "Error code" not in caplog.text
