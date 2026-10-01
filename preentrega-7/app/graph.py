import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.agents.analyst import analyst_node
from app.agents.researcher import DATA_DIR, researcher_node
from app.agents.guards import stall_guard
from app.agents.supervisor import supervisor_node
from app.hitl import IncidentChannel, publish_node
from app.schemas import Approval, Contribution, Evidence, Segment, ToolCallRecord, TraceStep
from app.state import OrchestratorState

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
    return "publish" if state["next_agent"] == "FINISH" else state["next_agent"]


def build_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver[Any],
    channel: IncidentChannel,
    data_dir: Path = DATA_DIR,
) -> Orchestrator:
    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", supervisor_node(llm), **stall_guard())
    graph.add_node("researcher", researcher_node(llm, data_dir))
    graph.add_node("analyst", analyst_node(llm))
    graph.add_node("publish", publish_node(channel))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", next_step, ["researcher", "analyst", "publish"])
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer)


type StepListener = Callable[[TraceStep], Awaitable[None]]


async def start_job(
    graph: Orchestrator,
    job_id: str,
    question: str,
    on_step: StepListener | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> Segment:
    first: OrchestratorState = {
        "messages": [HumanMessage(question)],
        "evidence": Evidence(),
        "delegations": [],
        "sender": "user",
    }
    return await _advance(graph, job_id, first, on_step, callbacks)


async def resume_job(
    graph: Orchestrator,
    job_id: str,
    approval: Approval,
    on_step: StepListener | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> Segment:
    return await _advance(graph, job_id, Command(resume=approval.model_dump()), on_step, callbacks)


async def _advance(
    graph: Orchestrator,
    job_id: str,
    payload: OrchestratorState | Command[Any],
    on_step: StepListener | None,
    callbacks: list[BaseCallbackHandler] | None,
) -> Segment:
    segment = Segment()
    tool_calls: list[ToolCallRecord] = []
    config: RunnableConfig = {
        "configurable": {"thread_id": job_id},
        "recursion_limit": RECURSION_LIMIT,
        "callbacks": callbacks or [],
    }
    async for part in graph.astream(payload, config, stream_mode="updates", subgraphs=True):
        namespace, update = cast(tuple[tuple[str, ...], dict[str, Any]], part)
        for node, output in update.items():
            if namespace:
                tool_calls.extend(_tool_calls(output))
                continue
            if node == "__interrupt__":
                segment.draft = output[0].value["draft"]
                continue
            segment.path.append(node)
            new_steps: list[TraceStep] = []
            if node == "supervisor":
                new_steps = list(output.get("delegations", []))
                if output["next_agent"] == "FINISH":
                    segment.final_answer = output["messages"][-1].text
                    segment.finish_reason = output["finish_reason"]
            elif node == "publish":
                segment.published = output["published"]
            else:
                new_steps = [Contribution(agent=node, content=output["messages"][-1].text, tool_calls=tool_calls)]
                tool_calls = []
            segment.steps.extend(new_steps)
            if on_step:
                for step in new_steps:
                    await on_step(step)
    return segment


def _tool_calls(output: dict[str, Any] | None) -> list[ToolCallRecord]:
    messages = (output or {}).get("messages", [])
    return [
        ToolCallRecord(name=call["name"], args=call["args"])
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
