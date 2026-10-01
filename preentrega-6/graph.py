import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import groq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst import analyst_node
from agents.researcher import DATA_DIR, researcher_node
from agents.specialist import stall_guard
from agents.supervisor import supervisor_node
from schemas import Contribution, Delegation, DelegationTrace, Evidence, ToolCallRecord
from state import OrchestratorState

DEFAULT_MODEL = "openai/gpt-oss-120b"
RECURSION_LIMIT = 12

type Orchestrator = CompiledStateGraph[OrchestratorState, None, OrchestratorState, OrchestratorState]


def create_llm() -> ChatGroq:
    return ChatGroq(
        model_name=os.getenv("GROQ_MODEL") or DEFAULT_MODEL,
        temperature=0,
        max_tokens=1536,
        reasoning_effort="low",
        request_timeout=30,
        max_retries=6,
    )


def next_step(state: OrchestratorState) -> str:
    return END if state["next_agent"] == "FINISH" else state["next_agent"]


def build_graph(llm: BaseChatModel, data_dir: Path = DATA_DIR) -> Orchestrator:
    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", supervisor_node(llm), **stall_guard())
    graph.add_node("researcher", researcher_node(llm, data_dir))
    graph.add_node("analyst", analyst_node(llm))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", next_step, ["researcher", "analyst", END])
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    return graph.compile()


type StepListener = Callable[[Delegation | Contribution], None]


async def run_investigation(
    graph: Orchestrator, question: str, on_step: StepListener | None = None
) -> DelegationTrace:
    trace = DelegationTrace(question=question)
    tool_calls: list[ToolCallRecord] = []
    updates = graph.astream(
        {"messages": [HumanMessage(question)], "evidence": Evidence(), "delegations": [], "sender": "user"},
        {"recursion_limit": RECURSION_LIMIT},
        stream_mode="updates",
        subgraphs=True,
    )
    try:
        async for part in updates:
            namespace, update = cast(tuple[tuple[str, ...], dict[str, Any]], part)
            for node, output in update.items():
                if namespace:
                    tool_calls.extend(_tool_calls(output))
                    continue
                trace.path.append(node)
                if node == "supervisor":
                    new_steps: list[Delegation | Contribution] = list(output.get("delegations", []))
                    if output["next_agent"] == "FINISH":
                        trace.final_answer = output["messages"][-1].text
                        trace.finish_reason = output["finish_reason"]
                else:
                    new_steps = [Contribution(agent=node, content=output["messages"][-1].text, tool_calls=tool_calls)]
                    tool_calls = []
                trace.steps.extend(new_steps)
                if on_step:
                    for step in new_steps:
                        on_step(step)
    except GraphRecursionError:
        trace.error = f"El grafo superó {RECURSION_LIMIT} pasos y se cortó el ciclo."
    except groq.APIError as error:
        trace.error = f"Groq falló: {error}"
    return trace


def _tool_calls(output: dict[str, Any] | None) -> list[ToolCallRecord]:
    messages = (output or {}).get("messages", [])
    return [
        ToolCallRecord(name=call["name"], args=call["args"])
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
