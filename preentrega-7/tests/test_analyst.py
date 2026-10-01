from typing import Any

import pytest
from langchain_core.tools import BaseTool

from app.agents.analyst import analyst_tools
from app.agents.researcher import DATA_DIR, read_log
from app.schemas import Evidence


@pytest.fixture
def tools() -> dict[str, BaseTool]:
    return {tool.name: tool for tool in analyst_tools()}


@pytest.fixture
def pool_incident() -> Evidence:
    return Evidence(log_lines=read_log(DATA_DIR / "log" / "pagos-api-2026-09-28.log"))


def run(tool: BaseTool, evidence: Evidence, **args: Any) -> str:
    message = tool.invoke(
        {"type": "tool_call", "name": tool.name, "args": {**args, "evidence": evidence}, "id": "call-1"}
    )
    return str(message.text)


def test_counts_lines_per_service_and_level_with_the_most_errors_first(
    tools: dict[str, BaseTool], pool_incident: Evidence
) -> None:
    result = run(tools["contar_por_servicio"], pool_incident)

    assert result.splitlines() == [
        "pagos-api: 4 ERROR, 1 WARN, 2 INFO",
        "pagos-worker: 1 ERROR, 1 WARN, 0 INFO",
        "pgbouncer: 0 ERROR, 1 WARN, 1 INFO",
        "postgres: 0 ERROR, 0 WARN, 1 INFO",
        "Total: 5 ERROR, 3 WARN, 4 INFO en 12 líneas",
    ]


def test_error_window_spans_from_the_first_signal_to_the_last_error(
    tools: dict[str, BaseTool], pool_incident: Evidence
) -> None:
    result = run(tools["ventana_de_errores"], pool_incident)

    assert result.splitlines() == [
        "Primera señal: 2026-09-28 21:14:05 WARN [pagos-api]",
        "Primer ERROR: 2026-09-28 21:14:09 [pagos-api]",
        "Último ERROR: 2026-09-28 21:14:20 [pagos-api]",
        "Duración con errores: 11 s",
        "Errores por minuto: 27.3",
    ]


def test_timeline_orders_lines_with_their_offset_from_the_first_one(
    tools: dict[str, BaseTool], pool_incident: Evidence
) -> None:
    result = run(tools["linea_de_tiempo"], pool_incident, nivel_minimo="WARN")

    lines = result.splitlines()
    assert lines[0] == "+0s WARN [pagos-api] pool de conexiones al 95% (19/20) en la réplica pagos-api-7c9f-2"
    assert lines[-1] == "+15s ERROR [pagos-api] POST /v1/payments -> 503 Service Unavailable comercio=4312"
    assert len(lines) == 8


def test_every_tool_says_there_is_no_evidence_instead_of_inventing_numbers(
    tools: dict[str, BaseTool],
) -> None:
    for tool in tools.values():
        assert run(tool, Evidence()).startswith("Sin evidencia")
