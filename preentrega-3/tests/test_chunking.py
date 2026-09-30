from pathlib import Path

import pytest

from indexer import load_chunks
from schemas import Category, ChunkingSettings

RUNBOOK = """\
# Cola de pagos caída

Intro breve del runbook.

## Síntomas

Los pagos quedan en estado pendiente y el consumer no avanza.

## Pasos

Reiniciar el consumer y revisar el lag de la cola.
"""


def write(data_dir: Path, relative: str, text: str) -> Path:
    path = data_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_markdown_chunks_carry_their_section(tmp_path: Path) -> None:
    path = write(tmp_path, "runbook/cola-caida.md", RUNBOOK)

    chunks = load_chunks(path, tmp_path, ChunkingSettings())

    sections = [chunk.metadata.section for chunk in chunks]
    assert sections == ["Cola de pagos caída", "Síntomas", "Pasos"]
    assert "Reiniciar el consumer" in chunks[2].text


def test_category_comes_from_the_folder_and_source_is_relative(tmp_path: Path) -> None:
    path = write(tmp_path, "log/gateway.log", "2026-09-28 ERROR 502 upstream\n")

    [chunk] = load_chunks(path, tmp_path, ChunkingSettings())

    assert chunk.metadata.category is Category.LOG
    assert chunk.metadata.source == "log/gateway.log"
    assert chunk.metadata.section == ""
    assert chunk.id == "log/gateway.log#0"


def test_long_text_is_split_with_overlap(tmp_path: Path) -> None:
    lines = [f"2026-09-28 14:{i:02d}:00 ERROR [pagos] timeout contra postgres" for i in range(40)]
    path = write(tmp_path, "log/pagos.log", "\n".join(lines))

    chunks = load_chunks(path, tmp_path, ChunkingSettings(chunk_size=300, chunk_overlap=60))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    assert [chunk.metadata.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[1].text.splitlines()[0] in chunks[0].text


def test_file_outside_a_category_folder_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "otros/notas.txt", "algo")

    with pytest.raises(ValueError, match="otros"):
        load_chunks(path, tmp_path, ChunkingSettings())


def test_empty_file_has_no_chunks(tmp_path: Path) -> None:
    path = write(tmp_path, "arquitectura/vacio.txt", "  \n")

    assert load_chunks(path, tmp_path, ChunkingSettings()) == []
