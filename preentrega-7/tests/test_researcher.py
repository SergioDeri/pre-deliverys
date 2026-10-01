from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from app.agents.researcher import DATA_DIR, researcher_tools
from app.schemas import Evidence


def tools_by_name(data_dir: Path = DATA_DIR) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in researcher_tools(data_dir)}


def call(tool: BaseTool, args: dict[str, Any]) -> ToolMessage:
    message = tool.invoke({"type": "tool_call", "name": tool.name, "args": args, "id": "call-1"})
    assert isinstance(message, ToolMessage)
    return message


@pytest.fixture
def tools() -> dict[str, BaseTool]:
    return tools_by_name()


def test_log_search_returns_the_lines_as_text_and_as_evidence(tools: dict[str, BaseTool]) -> None:
    message = call(tools["buscar_en_logs"], {"servicio": "pgbouncer", "fecha": "2026-09-28"})

    assert message.text.splitlines() == [
        "2026-09-28 21:14:11 WARN  [pgbouncer] cl_waiting=87 sv_active=200 pool=payflow",
        "2026-09-28 21:14:35 INFO  [pgbouncer] cl_waiting=0 sv_active=143 pool=payflow",
    ]
    evidence = message.artifact
    assert isinstance(evidence, Evidence)
    assert [(line.timestamp, line.level, line.service) for line in evidence.log_lines] == [
        (datetime(2026, 9, 28, 21, 14, 11), "WARN", "pgbouncer"),
        (datetime(2026, 9, 28, 21, 14, 35), "INFO", "pgbouncer"),
    ]
    assert evidence.error_codes == []


def test_error_code_lookup_returns_the_entry_as_evidence(tools: dict[str, BaseTool]) -> None:
    message = call(tools["buscar_codigo_error"], {"codigo": "pf-5021 "})

    assert "Timeout del procesador de tarjetas" in message.text
    assert "cola-de-pagos-caida" in message.text
    assert [entry.codigo for entry in message.artifact.error_codes] == ["PF-5021"]


def test_unknown_error_code_lists_the_valid_ones_and_adds_no_evidence(tools: dict[str, BaseTool]) -> None:
    message = call(tools["buscar_codigo_error"], {"codigo": "PF-9999"})

    assert message.text.startswith("Error:")
    assert "PF-5021" in message.text
    assert message.artifact == Evidence()


def test_runbook_is_read_by_the_name_the_catalog_gives(tools: dict[str, BaseTool]) -> None:
    message = call(tools["leer_runbook"], {"nombre": "runbook/base-de-datos-no-responde.md"})

    assert message.text.startswith("# ")
    assert "PgBouncer" in message.text
    assert message.artifact == Evidence()


def test_unknown_architecture_document_lists_the_available_ones(tools: dict[str, BaseTool]) -> None:
    message = call(tools["leer_arquitectura"], {"documento": "kubernetes"})

    assert message.text.startswith("Error:")
    assert "flujo-de-un-pago" in message.text
    assert "vision-general" in message.text


def test_log_search_filters_by_a_time_window(tools: dict[str, BaseTool]) -> None:
    message = call(
        tools["buscar_en_logs"], {"fecha": "2026-09-28", "desde": "21:14:10", "hasta": "21:14:13", "texto": "error"}
    )

    assert [str(line.timestamp.time()) for line in message.artifact.log_lines] == [
        "21:14:10",
        "21:14:13",
    ]


def test_time_window_also_accepts_full_timestamps(tools: dict[str, BaseTool]) -> None:
    message = call(
        tools["buscar_en_logs"], {"desde": "2026-09-28T21:14:35Z", "hasta": "2026-09-28 21:14:36"}
    )

    assert [line.service for line in message.artifact.log_lines] == ["pgbouncer", "pagos-api"]


def test_evidence_keeps_every_matching_line_even_when_the_text_is_cut(tmp_path: Path) -> None:
    (tmp_path / "log").mkdir()
    lines = [f"2026-09-28 21:{minute:02d}:00 ERROR [pagos-api] fallo {minute}" for minute in range(30)]
    (tmp_path / "log" / "pagos-api.log").write_text("\n".join(lines), encoding="utf-8")

    message = call(tools_by_name(tmp_path)["buscar_en_logs"], {"fecha": "2026-09-28"})

    assert len(message.text.splitlines()) == 21
    assert len(message.artifact.log_lines) == 30
