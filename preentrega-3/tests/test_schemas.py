import pytest
from pydantic import ValidationError

from schemas import Category, ChunkingSettings, SearchQuery


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        ChunkingSettings(chunk_size=200, chunk_overlap=200)


def test_default_chunking_is_800_with_120_of_overlap() -> None:
    assert ChunkingSettings() == ChunkingSettings(chunk_size=800, chunk_overlap=120)


def test_search_query_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        SearchQuery(text="   ")


@pytest.mark.parametrize("k", [0, 51])
def test_search_query_k_is_bounded(k: int) -> None:
    with pytest.raises(ValidationError):
        SearchQuery(text="timeout de base de datos", k=k)


def test_search_query_only_accepts_known_categories() -> None:
    assert SearchQuery(text="rollback", category="runbook").category is Category.RUNBOOK
    with pytest.raises(ValidationError):
        SearchQuery(text="rollback", category="manual")


def test_min_score_stays_within_cosine_range() -> None:
    with pytest.raises(ValidationError):
        SearchQuery(text="rollback", min_score=1.5)
