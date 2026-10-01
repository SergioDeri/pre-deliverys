import operator
from collections.abc import Awaitable
from typing import Annotated, Any, Literal, NotRequired, Protocol, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.schemas import Approval, Delegation, Evidence, Specialist

type NextAgent = Specialist | Literal["FINISH"]


def merge_evidence(current: Evidence, new: Evidence) -> Evidence:
    return Evidence(
        log_lines=list(dict.fromkeys([*current.log_lines, *new.log_lines])),
        error_codes=list(dict.fromkeys([*current.error_codes, *new.error_codes])),
    )


class OrchestratorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    evidence: Annotated[Evidence, merge_evidence]
    delegations: Annotated[list[Delegation], operator.add]
    sender: str
    next_agent: NotRequired[NextAgent]
    instruction: NotRequired[str]
    finish_reason: NotRequired[str]
    approval: NotRequired[Approval]
    published: NotRequired[bool]


class Node(Protocol):
    def __call__(self, state: OrchestratorState) -> Awaitable[dict[str, Any]]: ...


def question(state: OrchestratorState) -> str:
    return next(m.text for m in state["messages"] if isinstance(m, HumanMessage))


def final_answer(state: OrchestratorState) -> str:
    return next(m.text for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.name == "supervisor")


def contributions(state: OrchestratorState) -> list[AIMessage]:
    return [m for m in state["messages"] if isinstance(m, AIMessage) and m.name in ("researcher", "analyst")]


def team_contributions(state: OrchestratorState) -> str:
    return "\n\n".join(f"[{m.name}] {m.text}" for m in contributions(state)) or "Nadie aportó todavía."
