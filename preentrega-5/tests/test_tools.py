from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from tools import DATA_DIR, payflow_tools


def tools_by_name(data_dir: Path = DATA_DIR) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in payflow_tools(data_dir)}


@pytest.fixture
def tools() -> dict[str, BaseTool]:
    return tools_by_name()


def test_error_code_lookup_returns_cause_and_runbook(tools: dict[str, BaseTool]) -> None:
    result = tools["buscar_codigo_error"].invoke({"codigo": "PF-5021"})

    assert "Timeout del procesador de tarjetas" in result
    assert "Cobralia no respondió en 8 segundos" in result
    assert "cola-de-pagos-caida" in result


def test_unknown_error_code_lists_the_valid_ones(tools: dict[str, BaseTool]) -> None:
    result = tools["buscar_codigo_error"].invoke({"codigo": "PF-9999"})

    assert result.startswith("Error:")
    assert "PF-9999" in result
    assert "PF-5021" in result


def test_error_code_lookup_reports_a_corrupt_catalog(tmp_path: Path) -> None:
    (tmp_path / "errores").mkdir()
    (tmp_path / "errores" / "catalogo.json").write_text("[{", encoding="utf-8")

    result = tools_by_name(tmp_path)["buscar_codigo_error"].invoke({"codigo": "PF-5021"})

    assert result.startswith("Error:")
    assert "catálogo" in result


def test_error_code_is_normalized_before_lookup(tools: dict[str, BaseTool]) -> None:
    result = tools["buscar_codigo_error"].invoke({"codigo": " pf-5021 "})

    assert "Timeout del procesador de tarjetas" in result


def test_runbook_is_read_by_name(tools: dict[str, BaseTool]) -> None:
    result = tools["leer_runbook"].invoke({"nombre": "cola-de-pagos-caida"})

    assert result.startswith("# Runbook")


def test_unknown_runbook_lists_the_available_ones(tools: dict[str, BaseTool]) -> None:
    result = tools["leer_runbook"].invoke({"nombre": "reiniciar-todo"})

    assert result.startswith("Error:")
    assert "webhooks-no-llegan" in result


def test_runbook_name_cannot_escape_the_runbook_folder(tools: dict[str, BaseTool]) -> None:
    result = tools["leer_runbook"].invoke({"nombre": "../errores/catalogo.json"})

    assert result.startswith("Error:")
    assert "PF-5021" not in result


def test_log_search_matches_text_in_any_case(tools: dict[str, BaseTool]) -> None:
    result = tools["buscar_en_logs"].invoke({"texto": "pf-5021"})

    lines = result.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("2026-09-27 19:41:31 ERROR [pagos-worker] PF-5021")


def test_log_search_filters_by_the_service_tag_not_the_file(tools: dict[str, BaseTool]) -> None:
    result = tools["buscar_en_logs"].invoke({"servicio": "pagos-api", "fecha": "2026-09-29"})

    assert result == (
        "2026-09-29 10:04:02 ERROR [pagos-api] ImportError: cannot import name "
        "'ComercioSchema' from 'pagos.schemas'"
    )


def test_log_search_without_matches_suggests_services_and_dates(tools: dict[str, BaseTool]) -> None:
    result = tools["buscar_en_logs"].invoke({"texto": "PF-5021", "fecha": "2026-09-29"})

    assert not result.startswith("Error:")
    assert "Sin resultados" in result
    assert "pagos-worker" in result
    assert "2026-09-27" in result


def test_log_search_caps_the_result_and_says_so(tmp_path: Path) -> None:
    (tmp_path / "log").mkdir()
    noisy = "\n".join(f"2026-09-30 12:00:{second:02d} ERROR [pagos-api] PoolTimeout" for second in range(30))
    (tmp_path / "log" / "pagos-api-2026-09-30.log").write_text(noisy, encoding="utf-8")

    result = tools_by_name(tmp_path)["buscar_en_logs"].invoke({"texto": "PoolTimeout"})

    lines = result.splitlines()
    assert len(lines) == 21
    assert lines[19].startswith("2026-09-30 12:00:19")
    assert "30" in lines[20] and "20" in lines[20]


def test_log_search_reports_an_unreadable_log_folder(tmp_path: Path) -> None:
    result = tools_by_name(tmp_path)["buscar_en_logs"].invoke({"texto": "PF-5021"})

    assert result.startswith("Error:")


@pytest.mark.parametrize(
    "catalog",
    [
        '[{"titulo": "Sin código", "http": null, "causa": "x", "runbook": null}]',
        '{"codigo": "PF-5021"}',
    ],
)
def test_error_code_lookup_reports_a_malformed_catalog(tmp_path: Path, catalog: str) -> None:
    (tmp_path / "errores").mkdir()
    (tmp_path / "errores" / "catalogo.json").write_text(catalog, encoding="utf-8")

    result = tools_by_name(tmp_path)["buscar_codigo_error"].invoke({"codigo": "PF-5021"})

    assert result.startswith("Error:")


def test_error_code_lookup_reports_a_catalog_that_is_not_utf8(tmp_path: Path) -> None:
    (tmp_path / "errores").mkdir()
    (tmp_path / "errores" / "catalogo.json").write_bytes(b'[{"codigo": "PF-5021", "titulo": "\xff"}]')

    result = tools_by_name(tmp_path)["buscar_codigo_error"].invoke({"codigo": "PF-5021"})

    assert result.startswith("Error:")


def test_runbook_that_is_not_utf8_is_reported(tmp_path: Path) -> None:
    (tmp_path / "runbook").mkdir()
    (tmp_path / "runbook" / "roto.md").write_bytes(b"# Runbook \xff")

    result = tools_by_name(tmp_path)["leer_runbook"].invoke({"nombre": "roto"})

    assert result.startswith("Error:")
