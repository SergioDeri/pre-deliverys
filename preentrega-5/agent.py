import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, TypedDict

import groq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from schemas import Step, ToolCallRecord, ToolResult, TurnTrace

DEFAULT_MODEL = "openai/gpt-oss-120b"
CHECKPOINTS = Path(__file__).parent / "checkpoints.sqlite"
RECURSION_LIMIT = 10
SYSTEM_PROMPT = """\
Sos el asistente de guardia de PayFlow, una plataforma de pagos. Respondé en español, breve y \
al grano. Apoyate en las herramientas: el catálogo de errores, los runbooks y los logs de los \
servicios. No inventes códigos, causas, horarios ni comandos: citá solo los que leíste en ellas, \
y no muestres las herramientas como si fueran comandos para el usuario."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


type Agent = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def create_llm() -> ChatGroq:
    return ChatGroq(
        model_name=os.getenv("GROQ_MODEL") or DEFAULT_MODEL,
        temperature=0,
        max_tokens=2048,
        request_timeout=30,
        max_retries=2,
    )


def compile_agent(
    llm: BaseChatModel, tools: Sequence[BaseTool], checkpointer: BaseCheckpointSaver[Any]
) -> Agent:
    model = llm.bind_tools(tools)

    async def llm_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        reply = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")
    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_checkpointer(path: Path = CHECKPOINTS) -> AsyncIterator[AsyncSqliteSaver]:
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


async def thread_exists(checkpointer: BaseCheckpointSaver[Any], thread_id: str) -> bool:
    return await checkpointer.aget_tuple(thread_config(thread_id)) is not None


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


async def run_turn(agent: Agent, thread_id: str, question: str) -> TurnTrace:
    trace = TurnTrace(thread_id=thread_id, question=question)
    updates = agent.astream(
        {"messages": [HumanMessage(question)]}, thread_config(thread_id), stream_mode="updates"
    )
    try:
        async for update in updates:
            for node, output in update.items():
                step = _step(node, output["messages"])
                trace.steps.append(step)
                if step.content and not step.tool_calls:
                    trace.answer = step.content
    except GraphRecursionError:
        trace.error = f"El agente no llegó a una respuesta en {RECURSION_LIMIT} pasos y se cortó el ciclo."
    except groq.APIError as error:
        trace.error = f"Groq falló: {error}"
    return trace


def _step(node: str, messages: list[AnyMessage]) -> Step:
    step = Step(node=node)
    for message in messages:
        if isinstance(message, AIMessage):
            step.reasoning = message.additional_kwargs.get("reasoning_content")
            step.tool_calls = [ToolCallRecord(name=c["name"], args=c["args"]) for c in message.tool_calls]
            step.content = message.text or None
        elif isinstance(message, ToolMessage):
            step.tool_results.append(ToolResult(name=message.name or "", content=message.text))
    return step
