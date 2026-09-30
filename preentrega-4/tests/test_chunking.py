from pathlib import Path

import pytest
import tiktoken

from ingest import load_chunks, merge_sections
from schemas import Category, ChunkingSettings

RUNBOOK = """\
# Runbook: la cola no avanza

## Síntomas

El lag de pagos.autorizar supera los 1.000 mensajes.

## Mitigación

Reiniciar pagos-worker.
"""


def write(data_dir: Path, relative: str, text: str) -> Path:
    path = data_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_each_markdown_section_is_its_own_chunk(tmp_path: Path) -> None:
    path = write(tmp_path, "runbook/cola.md", RUNBOOK)

    chunks = load_chunks(path, tmp_path, ChunkingSettings())

    assert [chunk.metadata.section for chunk in chunks] == ["Síntomas", "Mitigación"]
    assert chunks[0].text.startswith("# Runbook: la cola no avanza")
    assert all(chunk.metadata.source == "runbook/cola.md" for chunk in chunks)
    assert all(chunk.metadata.category is Category.RUNBOOK for chunk in chunks)
    assert [chunk.id for chunk in chunks] == ["runbook/cola.md#0", "runbook/cola.md#1"]


def test_a_long_section_is_cut_to_the_token_budget(tmp_path: Path) -> None:
    body = " ".join(f"El worker {i} reintenta la transacción." for i in range(200))
    path = write(tmp_path, "arquitectura/largo.txt", body)
    encoding = tiktoken.get_encoding("cl100k_base")

    chunks = load_chunks(path, tmp_path, ChunkingSettings(chunk_tokens=100, overlap_tokens=20))

    assert len(chunks) > 1
    assert all(len(encoding.encode(chunk.text)) <= 100 for chunk in chunks)
    assert all(chunk.metadata.section == "" for chunk in chunks)


def test_each_error_code_is_one_chunk(tmp_path: Path) -> None:
    catalog = """[
      {"codigo": "PF-5030", "titulo": "Sin conexiones a la base", "http": 503,
       "causa": "PoolTimeout en pagos-api.", "runbook": "runbook/base.md"},
      {"codigo": "PF-4012", "titulo": "Comercio suspendido", "http": 403,
       "causa": "El comercio está suspendido.", "runbook": null}
    ]"""
    path = write(tmp_path, "errores/catalogo.json", catalog)

    chunks = load_chunks(path, tmp_path, ChunkingSettings())

    assert [chunk.metadata.section for chunk in chunks] == ["PF-5030", "PF-4012"]
    assert all(chunk.metadata.category is Category.ERRORES for chunk in chunks)
    assert "PF-5030" in chunks[0].text
    assert "PoolTimeout en pagos-api." in chunks[0].text
    assert "runbook/base.md" in chunks[0].text
    assert "Runbook:" not in chunks[1].text


def test_a_source_outside_a_category_folder_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "notas/sueltas.md", "# Notas\n\nalgo")

    with pytest.raises(ValueError, match="notas"):
        load_chunks(path, tmp_path, ChunkingSettings())


def test_merging_joins_consecutive_sections_up_to_the_token_budget(tmp_path: Path) -> None:
    path = write(tmp_path, "runbook/cola.md", RUNBOOK)
    sections = load_chunks(path, tmp_path, ChunkingSettings())

    merged = merge_sections(sections, ChunkingSettings())

    assert len(merged) == 1
    assert merged[0].metadata.section == "Síntomas"
    assert "Reiniciar pagos-worker." in merged[0].text
    assert merged[0].id == "runbook/cola.md#0"


def test_merging_never_goes_over_the_budget(tmp_path: Path) -> None:
    body = "\n\n".join(f"## Paso {i}\n\nEl worker {i} reintenta la transacción." for i in range(30))
    path = write(tmp_path, "runbook/pasos.md", body)
    encoding = tiktoken.get_encoding("cl100k_base")
    settings = ChunkingSettings(chunk_tokens=60, overlap_tokens=10)

    merged = merge_sections(load_chunks(path, tmp_path, settings), settings)

    assert 1 < len(merged) < 30
    assert all(len(encoding.encode(chunk.text)) <= 60 for chunk in merged)
    assert [chunk.metadata.chunk_index for chunk in merged] == list(range(len(merged)))
