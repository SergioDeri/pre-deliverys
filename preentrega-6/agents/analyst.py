from collections import Counter
from typing import Annotated, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from agents.specialist import compile_specialist, delegate
from schemas import Evidence, Level
from state import Node, OrchestratorState

PROMPT = """\
Sos el analista de un equipo que analiza incidentes de PayFlow, una plataforma de pagos. Trabajás \
sobre la evidencia que ya juntó el investigador: no tenés acceso a los logs ni a otras fuentes. \
Cualquier número que des (conteos, horarios, duraciones, tasas) tiene que salir de tus \
herramientas, nunca de hacer la cuenta vos. Mirá también la línea de tiempo completa, con las \
líneas INFO: a veces la causa o la recuperación aparece ahí y no en los errores. Con esos \
resultados, interpretá: qué servicio falló primero, cómo se propagó la falla, cuál parece la causa \
raíz y qué impacto tuvo, distinguiendo los tipos de error (no todos los ERROR son respuestas \
fallidas al comercio). Si la evidencia no alcanza para afirmar algo, decilo. Respondé en español, \
en pocas líneas."""

NO_EVIDENCE = "Sin evidencia: el investigador todavía no trajo líneas de log para analizar."
SEVERITY: dict[Level, int] = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}


def analyst_tools() -> list[BaseTool]:
    @tool
    def contar_por_servicio(evidence: Annotated[Evidence, InjectedState("evidence")]) -> str:
        """Cuenta las líneas de log de la evidencia por servicio y nivel (ERROR, WARN, INFO),
        con los servicios que más errores tuvieron primero."""
        if not evidence.log_lines:
            return NO_EVIDENCE
        counts = Counter((line.service, line.level) for line in evidence.log_lines)
        services = sorted(
            {line.service for line in evidence.log_lines},
            key=lambda service: (-counts[service, "ERROR"], -counts[service, "WARN"], service),
        )
        rows = [
            f"{service}: {counts[service, 'ERROR']} ERROR, {counts[service, 'WARN']} WARN, {counts[service, 'INFO']} INFO"
            for service in services
        ]
        totals = Counter(line.level for line in evidence.log_lines)
        rows.append(
            f"Total: {totals['ERROR']} ERROR, {totals['WARN']} WARN, {totals['INFO']} INFO "
            f"en {len(evidence.log_lines)} líneas"
        )
        return "\n".join(rows)

    @tool
    def ventana_de_errores(evidence: Annotated[Evidence, InjectedState("evidence")]) -> str:
        """Calcula cuándo empezó el incidente (primera línea WARN o ERROR), el primer y el último
        ERROR, cuánto duró el período con errores y cuántos errores hubo por minuto."""
        if not evidence.log_lines:
            return NO_EVIDENCE
        lines = sorted(evidence.log_lines, key=lambda line: line.timestamp)
        signals = [line for line in lines if SEVERITY[line.level] >= SEVERITY["WARN"]]
        errors = [line for line in lines if line.level == "ERROR"]
        if not errors:
            return f"No hay líneas ERROR en la evidencia ({len(signals)} WARN)."
        first, last = errors[0], errors[-1]
        seconds = (last.timestamp - first.timestamp).total_seconds()
        rate = f"{len(errors) / seconds * 60:.1f}" if seconds else "no aplica (un solo instante)"
        return "\n".join(
            [
                f"Primera señal: {signals[0].timestamp:%Y-%m-%d %H:%M:%S} {signals[0].level} [{signals[0].service}]",
                f"Primer ERROR: {first.timestamp:%Y-%m-%d %H:%M:%S} [{first.service}]",
                f"Último ERROR: {last.timestamp:%Y-%m-%d %H:%M:%S} [{last.service}]",
                f"Duración con errores: {seconds:.0f} s",
                f"Errores por minuto: {rate}",
            ]
        )

    @tool
    def linea_de_tiempo(
        evidence: Annotated[Evidence, InjectedState("evidence")],
        nivel_minimo: Literal["INFO", "WARN", "ERROR"] = "INFO",
        servicio: str | None = None,
    ) -> str:
        """Ordena las líneas de la evidencia en el tiempo, con los segundos transcurridos desde la
        primera. Sirve para ver qué servicio falló antes y cómo se propagó la falla."""
        if not evidence.log_lines:
            return NO_EVIDENCE
        lines = sorted(
            (
                line
                for line in evidence.log_lines
                if SEVERITY[line.level] >= SEVERITY[nivel_minimo]
                and (servicio is None or line.service == servicio.strip().lower())
            ),
            key=lambda line: line.timestamp,
        )
        if not lines:
            return f"Ninguna línea de nivel {nivel_minimo} o mayor" + (f" de {servicio}." if servicio else ".")
        start = lines[0].timestamp
        return "\n".join(
            f"+{(line.timestamp - start).total_seconds():.0f}s {line.level} [{line.service}] {line.message}"
            for line in lines
        )

    return [contar_por_servicio, ventana_de_errores, linea_de_tiempo]


def analyst_node(llm: BaseChatModel) -> Node:
    specialist = compile_specialist(llm, analyst_tools(), PROMPT)

    async def analyst(state: OrchestratorState) -> dict[str, Any]:
        messages = await delegate(specialist, "analyst", state, state["evidence"])
        return {"messages": [AIMessage(messages[-1].text, name="analyst")], "sender": "analyst"}

    return analyst
