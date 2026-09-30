from pathlib import Path

import chromadb
import pytest

from indexer import EmbeddingSpaceMismatch, Indexer, open_collection
from schemas import ChunkingSettings
from tests.fakes import FakeEmbedder

SMALL = ChunkingSettings(chunk_size=120, chunk_overlap=20)

LONG_LOG = "\n".join(
    f"2026-09-28 14:{i:02d}:00 ERROR [pagos] timeout contra postgres" for i in range(20)
)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    for relative, text in {
        "log/pagos.log": LONG_LOG,
        "runbook/rollback.md": "# Rollback\n\nVolver a la versión anterior con argo.",
        "arquitectura/notas.txt": "El gateway habla con el servicio de pagos.",
    }.items():
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return data


def make_indexer(tmp_path: Path, data_dir: Path, embedder: FakeEmbedder | None = None) -> Indexer:
    embedder = embedder or FakeEmbedder()
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = open_collection(client, "docs", embedder.space)
    return Indexer(collection, embedder, SMALL, data_dir)


def test_indexing_a_directory_stores_every_chunk(tmp_path: Path, data_dir: Path) -> None:
    indexer = make_indexer(tmp_path, data_dir)

    stored = indexer.index_directory()

    assert stored == indexer.count() > 3


def test_indexing_twice_does_not_duplicate(tmp_path: Path, data_dir: Path) -> None:
    indexer = make_indexer(tmp_path, data_dir)
    first = indexer.index_directory()

    indexer.index_directory()

    assert indexer.count() == first


def test_reindexing_a_shorter_source_leaves_no_orphan_chunks(
    tmp_path: Path, data_dir: Path
) -> None:
    indexer = make_indexer(tmp_path, data_dir)
    indexer.index_directory()
    before = indexer.count()
    log = data_dir / "log/pagos.log"
    old_chunks = indexer.index_source(log)

    log.write_text("2026-09-28 14:00:00 ERROR [pagos] timeout", encoding="utf-8")
    new_chunks = indexer.index_source(log)

    assert new_chunks == 1
    assert indexer.count() == before - old_chunks + 1


def test_deleting_a_source_removes_only_its_chunks(tmp_path: Path, data_dir: Path) -> None:
    indexer = make_indexer(tmp_path, data_dir)
    indexer.index_directory()
    before = indexer.count()

    deleted = indexer.delete_source("runbook/rollback.md")

    assert deleted == 1
    assert indexer.count() == before - 1
    assert indexer.delete_source("runbook/rollback.md") == 0


def test_index_survives_a_new_client(tmp_path: Path, data_dir: Path) -> None:
    stored = make_indexer(tmp_path, data_dir).index_directory()

    assert make_indexer(tmp_path, data_dir).count() == stored


def test_collection_refuses_a_different_embedding_space(tmp_path: Path, data_dir: Path) -> None:
    make_indexer(tmp_path, data_dir).index_directory()

    with pytest.raises(EmbeddingSpaceMismatch, match="fake-embedding"):
        make_indexer(tmp_path, data_dir, FakeEmbedder(dimension=32))
