from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from schemas import Delegation
from state import NextAgent, Node, OrchestratorState, contributions, question, team_contributions

DELEGATION_LIMIT = 4
DECISION_PROMPT = """\
Sos el supervisor de un equipo que investiga incidentes de PayFlow, una plataforma de pagos. No \
consultás datos vos: decidís quién trabaja a continuación.

- researcher: junta evidencia de los logs, el catálogo de errores, los runbooks y la arquitectura.
- analyst: calcula sobre la evidencia ya juntada (conteos por servicio, línea de tiempo, duración) \
y la interpreta. No puede buscar nada por su cuenta, así que solo sirve después del researcher.
- FINISH: ya hay suficiente para responder la pregunta del usuario.

Lo habitual es researcher, después analyst, después FINISH. Cualquier número de la respuesta \
(conteos, horarios, duraciones) tiene que venir del analyst: si la pregunta pide alguno y el \
analyst todavía no trabajó, no elijas FINISH. Volvé a un especialista solo si su aporte dejó algo \
concreto sin resolver. En instruction escribí una tarea concreta para el \
especialista elegido (qué buscar o qué calcular, con servicios y fechas si los sabés). Escribí \
instruction y reason en español."""
ANSWER_PROMPT = """\
Sos el supervisor de un equipo que investiga incidentes de PayFlow. Escribí la respuesta final \
para la persona de guardia a partir de los aportes del equipo: qué pasó, cuándo, qué servicio lo \
originó, con qué impacto y qué hacer. Usá solo datos que aparezcan en los aportes: los números \
tal como los dio el analista, y los pasos a seguir solo si vienen de un runbook que leyó el \
investigador, citándolo por su nombre. No inventes runbooks, comandos ni cifras; si algo no se \
pudo averiguar, decilo. Respondé en español, breve y al grano."""


class SupervisorDecision(BaseModel):
    """Quién trabaja a continuación, o FINISH si ya se puede responder."""

    next: NextAgent = Field(description="El especialista que trabaja ahora, o FINISH.")
    instruction: str = Field(description="Tarea concreta para el especialista. Vacía si next es FINISH.")
    reason: str = Field(description="Por qué elegiste ese paso, en una oración.")


def briefing(state: OrchestratorState) -> HumanMessage:
    return HumanMessage(
        f"Pregunta del usuario: {question(state)}\n\n"
        f"Delegaciones usadas: {len(state['delegations'])} de {DELEGATION_LIMIT}.\n\n"
        f"Aportes del equipo:\n{team_contributions(state)}"
    )


def supervisor_node(llm: BaseChatModel) -> Node:
    decider = llm.with_structured_output(SupervisorDecision)

    async def finish(state: OrchestratorState, reason: str) -> dict[str, Any]:
        answer = await llm.ainvoke([SystemMessage(ANSWER_PROMPT), briefing(state)])
        return {
            "messages": [AIMessage(answer.text, name="supervisor")],
            "next_agent": "FINISH",
            "finish_reason": reason,
            "sender": "supervisor",
        }

    async def supervisor(state: OrchestratorState) -> dict[str, Any]:
        if len(state["delegations"]) >= DELEGATION_LIMIT:
            return await finish(state, f"Se alcanzó el límite de {DELEGATION_LIMIT} delegaciones.")
        decision = SupervisorDecision.model_validate(
            await decider.ainvoke([SystemMessage(DECISION_PROMPT), briefing(state)])
        )
        if decision.next == "FINISH":
            return await finish(state, decision.reason)
        researched = any(m.name == "researcher" for m in contributions(state))
        override = decision.next == "analyst" and not researched
        delegation = Delegation(
            agent="researcher" if override else decision.next,
            instruction=decision.instruction,
            reason=decision.reason,
            override=override,
        )
        return {
            "next_agent": delegation.agent,
            "instruction": delegation.instruction,
            "delegations": [delegation],
            "sender": "supervisor",
        }

    return supervisor
