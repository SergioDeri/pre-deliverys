import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import SkipValidation

from app.schemas import Approval


@dataclass
class Stall:
    seconds: float
    reply: AIMessage = field(default_factory=lambda: AIMessage(content="tarde"))


class ScriptedChatModel(BaseChatModel):
    """Responde con los mensajes del guion en orden (o lanza la excepción que toque) y anota lo que recibió."""

    script: SkipValidation[list[AIMessage | Exception | Stall]]
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        reply = self.script[len(self.calls) - 1]
        if isinstance(reply, Stall):
            time.sleep(reply.seconds)
            return ChatResult(generations=[ChatGeneration(message=reply.reply)])
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def decide(next_agent: str, instruction: str = "", reason: str = "porque sí") -> AIMessage:
    args = {"next": next_agent, "instruction": instruction, "reason": reason}
    return tool_call("SupervisorDecision", args, f"decision-{next_agent}")


class RecordingChannel:
    def __init__(self) -> None:
        self.published: list[tuple[str, Approval]] = []

    async def publish(self, report: str, approval: Approval) -> None:
        self.published.append((report, approval))
