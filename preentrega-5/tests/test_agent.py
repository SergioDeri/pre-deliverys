from pathlib import Path

import groq
import httpx

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent import compile_agent, open_checkpointer, run_turn, thread_exists
from tests.fakes import ScriptedChatModel, tool_call
from tools import payflow_tools


async def test_turn_runs_the_tool_the_model_chose_and_answers_with_its_result() -> None:
    model = ScriptedChatModel(
        script=[
            tool_call("buscar_codigo_error", {"codigo": "PF-5021"}),
            AIMessage(content="PF-5021 es un timeout de Cobralia."),
        ]
    )
    agent = compile_agent(model, payflow_tools(), InMemorySaver())

    trace = await run_turn(agent, "guardia-1", "¿Qué es PF-5021?")

    assert [step.node for step in trace.steps] == ["llm", "tools", "llm"]
    assert trace.steps[0].tool_calls[0].name == "buscar_codigo_error"
    assert "Cobralia no respondió en 8 segundos" in trace.steps[1].tool_results[0].content
    assert "Cobralia no respondió en 8 segundos" in model.calls[1][-1].text
    assert trace.answer == "PF-5021 es un timeout de Cobralia."


async def test_thread_remembers_earlier_turns_after_reopening_the_checkpoint(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints.sqlite"
    async with open_checkpointer(checkpoints) as saver:
        first = ScriptedChatModel(script=[AIMessage(content="Anotado: PF-5021.")])
        await run_turn(compile_agent(first, payflow_tools(), saver), "guardia-1", "Estoy mirando PF-5021")

    async with open_checkpointer(checkpoints) as saver:
        second = ScriptedChatModel(script=[AIMessage(content="Me dijiste PF-5021."), AIMessage(content="No sé.")])
        agent = compile_agent(second, payflow_tools(), saver)
        await run_turn(agent, "guardia-1", "¿Qué código te dije?")
        await run_turn(agent, "guardia-2", "¿Qué código te dije?")

    resumed, fresh = second.calls
    assert [type(m) for m in resumed] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert resumed[1].text == "Estoy mirando PF-5021"
    assert all("PF-5021" not in m.text for m in fresh)


async def test_turn_that_never_stops_calling_tools_is_cut_at_the_recursion_limit() -> None:
    model = ScriptedChatModel(
        script=[tool_call("buscar_en_logs", {"texto": "PF-5021"}, f"call-{n}") for n in range(20)]
    )
    agent = compile_agent(model, payflow_tools(), InMemorySaver())

    trace = await run_turn(agent, "guardia-1", "Buscá hasta encontrar algo")

    assert trace.answer is None
    assert trace.error is not None and "10" in trace.error
    assert len(trace.steps) == 10


async def test_model_sees_invalid_tool_arguments_as_an_error_and_can_retry() -> None:
    model = ScriptedChatModel(
        script=[
            tool_call("buscar_codigo_error", {"codigo": "5021"}, "call-1"),
            tool_call("buscar_codigo_error", {"codigo": "PF-5021"}, "call-2"),
            AIMessage(content="PF-5021 es un timeout de Cobralia."),
        ]
    )
    agent = compile_agent(model, payflow_tools(), InMemorySaver())

    trace = await run_turn(agent, "guardia-1", "¿Qué es el 5021?")

    assert "PF-" in trace.steps[1].tool_results[0].content
    assert "Cobralia no respondió" in trace.steps[3].tool_results[0].content
    assert trace.answer == "PF-5021 es un timeout de Cobralia."


async def test_provider_failure_mid_turn_keeps_the_steps_already_taken() -> None:
    outage = groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
    model = ScriptedChatModel(script=[tool_call("buscar_codigo_error", {"codigo": "PF-5021"}), outage])
    agent = compile_agent(model, payflow_tools(), InMemorySaver())

    trace = await run_turn(agent, "guardia-1", "¿Qué es PF-5021?")

    assert [step.node for step in trace.steps] == ["llm", "tools"]
    assert trace.answer is None
    assert trace.error is not None and "Groq" in trace.error


async def test_only_threads_with_a_saved_turn_exist(tmp_path: Path) -> None:
    async with open_checkpointer(tmp_path / "checkpoints.sqlite") as saver:
        model = ScriptedChatModel(script=[AIMessage(content="Hola.")])
        await run_turn(compile_agent(model, payflow_tools(), saver), "guardia-1", "Hola")

        assert await thread_exists(saver, "guardia-1")
        assert not await thread_exists(saver, "guardia-l")
