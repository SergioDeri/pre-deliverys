from pathlib import Path

import chromadb
import pytest
from chromadb.api.models.Collection import Collection

from indexer import Indexer, open_collection
from retriever import Retriever
from schemas import Category, ChunkingSettings, SearchQuery
from tests.fakes import FakeEmbedder

DOCS = {
    "log/pagos.log": "ERROR pagos timeout contra la base de datos postgres",
    "runbook/base-de-datos.md": "# Base de datos lenta\n\nRevisar conexiones de postgres y el pool.",
    "runbook/rollback.md": "# Rollback\n\nVolver a la versión anterior del deploy con argo.",
    "arquitectura/gateway.txt": "El gateway recibe las peticiones y las manda al servicio de pagos.",
}


@pytest.fixture
def collection(tmp_path: Path) -> Collection:
    data = tmp_path / "data"
    for relative, text in DOCS.items():
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    embedder = FakeEmbedder()
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = open_collection(client, "docs", embedder.space)
    Indexer(collection, embedder, ChunkingSettings(), data).index_directory()
    return collection


def test_closest_chunk_comes_first_with_its_metadata(collection: Collection) -> None:
    retriever = Retriever(collection, FakeEmbedder())

    results = retriever.search(SearchQuery(text="timeout de la base de datos postgres", k=2))

    assert len(results) == 2
    assert results[0].metadata.source == "log/pagos.log"
    assert results[0].metadata.category is Category.LOG
    assert results[0].score >= results[1].score
    assert results[0].score == pytest.approx(1 - results[0].distance)


def test_identical_text_scores_one(collection: Collection) -> None:
    text = DOCS["arquitectura/gateway.txt"]

    [result] = Retriever(collection, FakeEmbedder()).search(SearchQuery(text=text, k=1))

    assert result.text == text
    assert result.score == pytest.approx(1.0, abs=1e-4)


def test_category_filter_keeps_only_that_category(collection: Collection) -> None:
    query = SearchQuery(text="timeout de la base de datos postgres", k=10, category="runbook")

    results = Retriever(collection, FakeEmbedder()).search(query)

    assert {r.metadata.source for r in results} == {
        "runbook/base-de-datos.md",
        "runbook/rollback.md",
    }
    assert results[0].metadata.source == "runbook/base-de-datos.md"


def test_source_filter_keeps_only_that_source(collection: Collection) -> None:
    query = SearchQuery(text="postgres", k=10, source="log/pagos.log")

    results = Retriever(collection, FakeEmbedder()).search(query)

    assert [r.metadata.source for r in results] == ["log/pagos.log"]


def test_category_and_source_must_both_match(collection: Collection) -> None:
    query = SearchQuery(text="postgres", k=10, category="runbook", source="log/pagos.log")

    assert Retriever(collection, FakeEmbedder()).search(query) == []


def test_min_score_drops_distant_results(collection: Collection) -> None:
    text = DOCS["runbook/rollback.md"]
    query = SearchQuery(text=text, k=10, min_score=0.9)

    results = Retriever(collection, FakeEmbedder()).search(query)

    assert [r.metadata.source for r in results] == ["runbook/rollback.md"]


def test_empty_collection_returns_nothing(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = open_collection(client, "vacia", embedder.space)

    assert Retriever(collection, embedder).search(SearchQuery(text="postgres")) == []
