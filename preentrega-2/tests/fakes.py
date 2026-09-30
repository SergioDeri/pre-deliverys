from collections.abc import Sequence
from typing import Any

import groq
import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

VALID_ARGS: dict[str, Any] = {
    "tipo_de_entrada": "log_error",
    "tecnologias": ["PostgreSQL", "FastAPI"],
    "componentes_afectados": ["api-pedidos"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "El pool de conexiones a PostgreSQL se agota y la API responde 503.",
}


def tool_reply(args: dict[str, Any], finish_reason: str = "tool_calls") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "TechnicalAnalysis", "args": args, "id": "call_1"}],
        response_metadata={"finish_reason": finish_reason},
    )


def status_error(
    cls: type[groq.APIStatusError], status: int, message: str = "boom", code: str | None = None
) -> groq.APIStatusError:
    response = httpx.Response(status, request=REQUEST)
    body = {"error": {"message": message, "type": "invalid_request_error", "code": code}}
    return cls(f"Error code: {status} - {body}", response=response, body=body)


def tool_use_failed() -> groq.APIStatusError:
    return status_error(groq.BadRequestError, 400, "Failed to call a function.", "tool_use_failed")


class ScriptedChatModel(BaseChatModel):
    replies: list[Any]
    received: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(messages)
        reply = self.replies[min(len(self.received), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])
