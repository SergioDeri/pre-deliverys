import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from evaluate import check_sources, precision_at_k, rank, recall_at_k, retrieved_sources, score
from schemas import GoldenCase


def chunk_of(source: str) -> Document:
    return Document(page_content=f"texto de {source}", metadata={"source": source})


def test_retrieved_sources_keep_the_first_appearance_of_each_source() -> None:
    ranking = [chunk_of(s) for s in ["a.md", "b.md", "a.md", "c.md", "b.md"]]

    assert retrieved_sources(ranking) == ["a.md", "b.md", "c.md"]


def test_precision_counts_expected_sources_among_the_first_k() -> None:
    retrieved = ["a.md", "x.md", "b.md", "y.md", "z.md"]

    assert precision_at_k(retrieved, {"a.md", "b.md"}, k=3) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, {"a.md", "b.md"}, k=5) == pytest.approx(2 / 5)


def test_precision_divides_by_k_even_when_fewer_sources_come_back() -> None:
    assert precision_at_k(["a.md"], {"a.md"}, k=3) == pytest.approx(1 / 3)


def test_recall_is_the_share_of_expected_sources_found_in_the_first_k() -> None:
    retrieved = ["a.md", "x.md", "y.md", "b.md"]

    assert recall_at_k(retrieved, {"a.md", "b.md"}, k=3) == pytest.approx(1 / 2)
    assert recall_at_k(retrieved, {"a.md", "b.md"}, k=5) == 1.0


def test_nothing_retrieved_scores_zero() -> None:
    assert precision_at_k([], {"a.md"}, k=3) == 0.0
    assert recall_at_k([], {"a.md"}, k=3) == 0.0


def case(question: str, *expected: str) -> GoldenCase:
    return GoldenCase.model_validate({"pregunta": question, "documentos_esperados": list(expected)})


def test_score_averages_every_question_at_each_k() -> None:
    rankings = {
        "rollback": ["runbook/rollback.md", "log/gateway.log", "arquitectura/vision.md"],
        "argo": ["log/gateway.log", "arquitectura/vision.md", "runbook/otro.md"],
    }
    retriever: RunnableLambda[str, list[Document]] = RunnableLambda(
        lambda question: [chunk_of(s) for s in rankings[question]]
    )
    cases = [
        case("rollback", "runbook/rollback.md"),
        case("argo", "arquitectura/vision.md", "runbook/rollback.md"),
    ]

    scores = score(rank(retriever, cases), cases, cutoffs=(1, 3))

    assert [s.k for s in scores] == [1, 3]
    assert scores[0].precision == pytest.approx((1 + 0) / 2)
    assert scores[0].recall == pytest.approx((1 + 0) / 2)
    assert scores[1].precision == pytest.approx((1 / 3 + 1 / 3) / 2)
    assert scores[1].recall == pytest.approx((1 + 1 / 2) / 2)


def test_expected_sources_missing_from_the_index_are_reported() -> None:
    cases = [case("rollback", "runbook/rollback.md", "runbook/rolback.md")]

    with pytest.raises(ValueError, match="runbook/rolback.md"):
        check_sources(cases, indexed={"runbook/rollback.md"})
