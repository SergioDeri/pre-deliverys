import asyncio
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import groq
from langchain_core.language_models import BaseChatModel
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables.retry import ExponentialJitterParams
from langchain_groq import ChatGroq
from pydantic import SecretStr, ValidationError

from prompts import EXTRACTION_PROMPT, correction_messages
from schemas import ExtractionFailure, ExtractionOutcome, TechnicalAnalysis

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BACKOFF = ExponentialJitterParams(initial=2, max=20, jitter=1)

_TRANSIENT_PROVIDER_ERRORS = (
    groq.RateLimitError,
    groq.InternalServerError,
    groq.APIConnectionError,
)

type ExtractionChain = Runnable[dict[str, Any], TechnicalAnalysis]


class InvalidOutputError(Exception):
    pass


class TruncatedOutputError(Exception):
    pass


def create_llm(api_key: str | None = None) -> ChatGroq:
    # Retries belong to the chain (see build_chain); the SDK's would hide them and double the wait.
    return ChatGroq(
        model_name=os.getenv("GROQ_MODEL") or DEFAULT_MODEL,
        groq_api_key=SecretStr(api_key) if api_key else None,
        temperature=0,
        max_tokens=1024,
        request_timeout=30,
        max_retries=0,
    )


def build_chain(
    llm: BaseChatModel,
    *,
    max_attempts: int = 3,
    backoff: ExponentialJitterParams = DEFAULT_BACKOFF,
) -> ExtractionChain:
    structured = llm.with_structured_output(TechnicalAnalysis, include_raw=True)
    attempt = (
        RunnableLambda(_add_correction)
        | EXTRACTION_PROMPT
        | _reporting_failed_calls(structured)
        | RunnableLambda(_accept)
    ).with_retry(
        retry_if_exception_type=_TRANSIENT_PROVIDER_ERRORS,
        exponential_jitter_params=backoff,
        stop_after_attempt=max_attempts,
    )
    # At temperature 0 a plain retry repeats the same answer, so a rejected reply is
    # retried with the reason attached: the fallback receives it under previous_error.
    return attempt.with_fallbacks(
        [attempt] * (max_attempts - 1),
        exceptions_to_handle=(InvalidOutputError, TruncatedOutputError),
        exception_key="previous_error",
    )


def _add_correction(inputs: dict[str, Any]) -> dict[str, Any]:
    error = inputs.get("previous_error")
    return inputs | {"correction": correction_messages(error) if error else []}


def _reporting_failed_calls(
    structured: Runnable[PromptValue, Any],
) -> Runnable[PromptValue, dict[str, Any]]:
    async def call(prompt: PromptValue, config: RunnableConfig) -> dict[str, Any]:
        try:
            result: dict[str, Any] = await structured.ainvoke(prompt, config)
        except _TRANSIENT_PROVIDER_ERRORS as exc:
            raise _rejected(exc)
        except groq.BadRequestError as exc:
            # Groq answers 400 when the model produced a tool call it could not parse:
            # that is a malformed reply, not a bad request, so it is worth another try.
            if _error_code(exc) == "tool_use_failed":
                raise _rejected(InvalidOutputError(_provider_message(exc))) from exc
            raise
        return result

    return RunnableLambda(call)


def _error_detail(exc: groq.APIError) -> dict[str, Any]:
    detail = exc.body.get("error") if isinstance(exc.body, dict) else None
    return detail if isinstance(detail, dict) else {}


def _error_code(exc: groq.APIError) -> object:
    return _error_detail(exc).get("code")


def _provider_message(exc: groq.APIError) -> str:
    text = str(_error_detail(exc).get("message") or exc.message)
    status = getattr(exc, "status_code", None)
    return f"HTTP {status}: {text}" if status else text


def _describe(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'análisis'}: {e['msg']}" for e in error.errors()
        )
    return str(error)


def _accept(result: dict[str, Any]) -> TechnicalAnalysis:
    if result["raw"].response_metadata.get("finish_reason") == "length":
        raise _rejected(TruncatedOutputError("la respuesta se cortó por el límite de tokens"))
    error = result["parsing_error"]
    if error is not None:
        raise _rejected(InvalidOutputError(_describe(error)))
    parsed = result["parsed"]
    if not isinstance(parsed, TechnicalAnalysis):
        raise _rejected(InvalidOutputError("el modelo no devolvió ningún análisis"))
    return parsed


def _rejected(error: Exception) -> Exception:
    detail = _provider_message(error) if isinstance(error, groq.APIError) else str(error)
    logger.warning("intento descartado (%s): %s", type(error).__name__, detail)
    return error


async def extract(chain: ExtractionChain, source_text: str) -> ExtractionOutcome:
    started = time.perf_counter()
    analysis = failure = None
    try:
        analysis = await chain.ainvoke({"input_text": source_text})
    except InvalidOutputError as exc:
        failure = ExtractionFailure(kind="invalid_output", message=str(exc))
    except TruncatedOutputError as exc:
        failure = ExtractionFailure(kind="truncated_output", message=str(exc))
    except groq.APIError as exc:
        failure = ExtractionFailure(kind="provider", message=_provider_message(exc))
    return ExtractionOutcome(
        analysis=analysis,
        failure=failure,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


async def extract_many(
    chain: ExtractionChain, source_texts: Sequence[str]
) -> list[ExtractionOutcome]:
    return await asyncio.gather(*(extract(chain, text) for text in source_texts))
