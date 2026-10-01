from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.errors import NodeTimeoutError
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy, TimeoutPolicy

from schemas import Evidence, Specialist
from state import OrchestratorState, question, team_contributions

MAX_TOOL_ROUNDS = 5
RECURSION_LIMIT = 2 * MAX_TOOL_ROUNDS + 3
STEP_TIMEOUT = 120.0
OUT_OF_ROUNDS = (
    f"Ya usaste tus {MAX_TOOL_ROUNDS} rondas de herramientas. Respondé ahora con lo que tenés "
    "y decí qué te faltó averiguar."
)


def stall_guard() -> dict[str, Any]:
    return {
        "timeout": TimeoutPolicy(run_timeout=STEP_TIMEOUT),
        "retry_policy": RetryPolicy(max_attempts=2, retry_on=NodeTimeoutError),
    }


class SpecialistState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    evidence: Evidence


type SpecialistGraph = CompiledStateGraph[SpecialistState, None, SpecialistState, SpecialistState]


def compile_specialist(llm: BaseChatModel, tools: Sequence[BaseTool], prompt: str) -> SpecialistGraph:
    with_tools = llm.bind_tools(tools)

    async def think(state: SpecialistState) -> dict[str, list[AnyMessage]]:
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        if rounds >= MAX_TOOL_ROUNDS:
            messages = [SystemMessage(prompt), *state["messages"], HumanMessage(OUT_OF_ROUNDS)]
            return {"messages": [await llm.ainvoke(messages)]}
        return {"messages": [await with_tools.ainvoke([SystemMessage(prompt), *state["messages"]])]}

    graph = StateGraph(SpecialistState)
    graph.add_node("think", think, **stall_guard())
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "think")
    graph.add_conditional_edges("think", tools_condition)
    graph.add_edge("tools", "think")
    return graph.compile()


def assignment(state: OrchestratorState) -> str:
    return (
        f"Pregunta del usuario: {question(state)}\n\n"
        f"Aportes previos del equipo:\n{team_contributions(state)}\n\n"
        f"Tu tarea: {state['instruction']}"
    )


async def delegate(
    specialist: SpecialistGraph, name: Specialist, state: OrchestratorState, evidence: Evidence
) -> list[AnyMessage]:
    result = await specialist.ainvoke(
        {"messages": [HumanMessage(assignment(state))], "evidence": evidence},
        {"recursion_limit": RECURSION_LIMIT, "run_name": name},
    )
    messages: list[AnyMessage] = result["messages"]
    return messages
