import groq
import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import app.agents.guards
from app.graph import StepListener, build_graph, resume_job, start_job
from app.schemas import Approval, Contribution, Delegation, Segment
from tests.fakes import RecordingChannel, ScriptedChatModel, Stall, decide, tool_call

QUESTION = "¿Qué pasó el 28/09 a la noche con los pagos?"


async def investigate(
    model: ScriptedChatModel,
    channel: RecordingChannel | None = None,
    on_step: StepListener | None = None,
) -> Segment:
    graph = build_graph(model, InMemorySaver(), channel or RecordingChannel())
    return await start_job(graph, "job-1", QUESTION, on_step=on_step)


async def test_supervisor_delegates_to_the_researcher_then_the_analyst_then_answers() -> None:
    model = ScriptedChatModel(
        script=[
            decide("researcher", "Traé los logs del 28/09"),
            tool_call("buscar_en_logs", {"fecha": "2026-09-28"}),
            AIMessage(content="Hay 12 líneas del 28/09 entre 21:14:02 y 21:14:36."),
            decide("analyst", "Contá los errores por servicio"),
            tool_call("contar_por_servicio", {}),
            AIMessage(content="pagos-api concentra 4 de los 5 errores."),
            decide("FINISH", reason="Ya hay evidencia y análisis"),
            AIMessage(content="El pool de pagos-api se agotó por una consulta de 312 s."),
        ]
    )

    heard: list[Delegation | Contribution] = []

    async def listen(step: Delegation | Contribution) -> None:
        heard.append(step)

    trace = await investigate(model, on_step=listen)

    assert trace.path == ["supervisor", "researcher", "supervisor", "analyst", "supervisor"]
    assert heard == trace.steps
    assert trace.final_answer == "El pool de pagos-api se agotó por una consulta de 312 s."
    assert trace.finish_reason == "Ya hay evidencia y análisis"
    delegations = [step for step in trace.steps if isinstance(step, Delegation)]
    assert [(d.agent, d.instruction) for d in delegations] == [
        ("researcher", "Traé los logs del 28/09"),
        ("analyst", "Contá los errores por servicio"),
    ]
    contributions = [step for step in trace.steps if isinstance(step, Contribution)]
    assert [c.agent for c in contributions] == ["researcher", "analyst"]
    assert contributions[0].tool_calls[0].name == "buscar_en_logs"
    assert "Total: 5 ERROR, 3 WARN, 4 INFO en 12 líneas" in model.calls[5][-1].text


async def test_specialists_hand_back_only_their_contribution() -> None:
    model = ScriptedChatModel(
        script=[
            decide("researcher", "Traé los logs del 28/09"),
            tool_call("buscar_en_logs", {"fecha": "2026-09-28"}),
            AIMessage(content="Hay 12 líneas del 28/09."),
            decide("analyst", "Contá los errores por servicio"),
            AIMessage(content="Listo."),
            decide("FINISH"),
            AIMessage(content="Respuesta."),
        ]
    )

    await investigate(model)

    supervisor_input = " ".join(m.text for m in model.calls[3])
    analyst_input = " ".join(m.text for m in model.calls[4])
    assert "Hay 12 líneas del 28/09." in supervisor_input
    assert "Hay 12 líneas del 28/09." in analyst_input
    for seen in (supervisor_input, analyst_input):
        assert "cl_waiting=87" not in seen


async def test_analyst_chosen_before_any_research_goes_to_the_researcher_as_an_override() -> None:
    model = ScriptedChatModel(
        script=[
            decide("analyst", "Calculá la duración del incidente"),
            AIMessage(content="No encontré nada todavía."),
            decide("FINISH"),
            AIMessage(content="No hay datos suficientes."),
        ]
    )

    trace = await investigate(model)

    assert trace.path == ["supervisor", "researcher", "supervisor"]
    assert trace.steps[0] == Delegation(
        agent="researcher", instruction="Calculá la duración del incidente", reason="porque sí", override=True
    )


async def test_supervisor_must_answer_once_the_delegation_limit_is_reached() -> None:
    model = ScriptedChatModel(
        script=[
            *[msg for n in range(4) for msg in (decide("researcher", f"Buscá más ({n})"), AIMessage(content="Nada."))],
            AIMessage(content="Respuesta con lo que hay."),
        ]
    )

    trace = await investigate(model)

    assert trace.path == ["supervisor", "researcher"] * 4 + ["supervisor"]
    assert trace.final_answer == "Respuesta con lo que hay."
    assert trace.finish_reason is not None and "4" in trace.finish_reason
    assert len(model.calls) == 9


async def test_evidence_found_twice_is_counted_once() -> None:
    model = ScriptedChatModel(
        script=[
            decide("researcher", "Logs del 28/09"),
            tool_call("buscar_en_logs", {"fecha": "2026-09-28"}),
            AIMessage(content="12 líneas."),
            decide("researcher", "Logs de pagos-api del 28/09"),
            tool_call("buscar_en_logs", {"fecha": "2026-09-28", "servicio": "pagos-api"}),
            AIMessage(content="7 líneas de pagos-api."),
            decide("analyst", "Contá"),
            tool_call("contar_por_servicio", {}),
            AIMessage(content="Contado."),
            decide("FINISH"),
            AIMessage(content="Respuesta."),
        ]
    )

    await investigate(model)

    assert "Total: 5 ERROR, 3 WARN, 4 INFO en 12 líneas" in model.calls[8][-1].text


async def test_specialist_out_of_tool_rounds_answers_with_what_it_has() -> None:
    searches = [tool_call("buscar_en_logs", {"texto": f"intento {n}"}, f"call-{n}") for n in range(5)]
    model = ScriptedChatModel(
        script=[
            decide("researcher", "Buscá hasta encontrar algo"),
            *searches,
            AIMessage(content="No encontré nada en cinco búsquedas."),
            decide("FINISH"),
            AIMessage(content="Respuesta."),
        ]
    )

    trace = await investigate(model)

    assert trace.path == ["supervisor", "researcher", "supervisor"]
    contribution = next(step for step in trace.steps if isinstance(step, Contribution))
    assert contribution.content == "No encontré nada en cinco búsquedas."
    assert len(contribution.tool_calls) == 5


async def test_a_model_call_that_hangs_is_abandoned_and_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app.agents.guards, "STEP_TIMEOUT", 0.2)
    model = ScriptedChatModel(
        script=[
            Stall(1.0),
            decide("researcher", "Logs del 28/09"),
            Stall(1.0),
            AIMessage(content="Nada."),
            decide("FINISH"),
            AIMessage(content="Respuesta."),
        ]
    )

    trace = await investigate(model)

    assert trace.path == ["supervisor", "researcher", "supervisor"]
    assert trace.final_answer == "Respuesta."


FINISHED_INVESTIGATION = [
    decide("researcher", "Logs del 28/09"),
    AIMessage(content="12 líneas."),
    decide("FINISH"),
    AIMessage(content="El pool de pagos-api se agotó."),
]


async def test_finished_investigation_pauses_with_the_final_answer_as_the_draft() -> None:
    channel = RecordingChannel()

    segment = await investigate(ScriptedChatModel(script=list(FINISHED_INVESTIGATION)), channel)

    assert segment.draft == "El pool de pagos-api se agotó."
    assert segment.published is None
    assert channel.published == []


async def test_approval_resumes_the_job_and_publishes_the_draft() -> None:
    channel = RecordingChannel()
    graph = build_graph(ScriptedChatModel(script=list(FINISHED_INVESTIGATION)), InMemorySaver(), channel)
    await start_job(graph, "job-1", QUESTION)

    segment = await resume_job(graph, "job-1", Approval(approved=True, approver="ana", comment="ok"))

    assert segment.path == ["publish"]
    assert segment.draft is None
    assert segment.published is True
    assert [(report, approval.approver) for report, approval in channel.published] == [
        ("El pool de pagos-api se agotó.", "ana")
    ]


async def test_rejection_finishes_the_job_without_publishing() -> None:
    channel = RecordingChannel()
    graph = build_graph(ScriptedChatModel(script=list(FINISHED_INVESTIGATION)), InMemorySaver(), channel)
    await start_job(graph, "job-1", QUESTION)

    rejection = Approval(approved=False, approver="ana", comment="falta la causa")
    segment = await resume_job(graph, "job-1", rejection)

    assert segment.published is False
    assert channel.published == []


async def test_a_paused_job_resumes_from_its_checkpoint_in_a_new_graph() -> None:
    checkpoints = InMemorySaver()
    channel = RecordingChannel()
    first = build_graph(ScriptedChatModel(script=list(FINISHED_INVESTIGATION)), checkpoints, channel)
    await start_job(first, "job-1", QUESTION)

    restarted = build_graph(ScriptedChatModel(script=[]), checkpoints, channel)
    segment = await resume_job(restarted, "job-1", Approval(approved=True, approver="ana"))

    assert segment.published is True
    assert channel.published[0][0] == "El pool de pagos-api se agotó."


def rate_limited() -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, json={"error": {"message": "tokens per minute"}})
    return groq.RateLimitError("Rate limit reached", response=response, body=None)


async def test_a_rate_limited_model_call_waits_and_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app.agents.guards, "RATE_LIMIT_WAIT", 0.01)
    model = ScriptedChatModel(
        script=[
            rate_limited(),
            decide("researcher", "Logs del 28/09"),
            rate_limited(),
            rate_limited(),
            AIMessage(content="Nada."),
            decide("FINISH"),
            AIMessage(content="Respuesta."),
        ]
    )

    segment = await investigate(model)

    assert segment.path == ["supervisor", "researcher", "supervisor"]
    assert segment.draft == "Respuesta."
