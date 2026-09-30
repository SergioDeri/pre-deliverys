import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from hybrid_retriever import HybridRetrieval, tokenize
from schemas import Category, HybridWeights
from tests.fakes import FakeEmbeddings

CHUNKS = [
    Document(
        page_content="PF-5030: sin conexiones disponibles a la base, PoolTimeout en pagos-api",
        metadata={"source": "errores/catalogo.json", "category": "errores"},
    ),
    Document(
        page_content="Mitigación: cancelar las consultas largas y bajar réplicas de pagos-worker",
        metadata={"source": "runbook/base.md", "category": "runbook"},
    ),
    Document(
        page_content="RabbitMQ reentrega el mensaje cuando la base no responde",
        metadata={"source": "arquitectura/flujo.md", "category": "arquitectura"},
    ),
    Document(
        page_content="ERROR pagos-api PoolTimeout: couldn't get a connection after 5 sec",
        metadata={"source": "log/pagos-api.log", "category": "log"},
    ),
]


@pytest.fixture
def retrieval() -> HybridRetrieval:
    embeddings = FakeEmbeddings()
    stores = {}
    for category in Category:
        store = InMemoryVectorStore(embeddings)
        docs = [doc for doc in CHUNKS if doc.metadata["category"] == category]
        if docs:
            store.add_documents(docs)
        stores[category] = store
    return HybridRetrieval(stores, CHUNKS, fetch_k=4)


def sources(docs: list[Document]) -> list[str]:
    return [doc.metadata["source"] for doc in docs]


def test_tokenize_folds_case_and_accents_and_keeps_codes_whole() -> None:
    assert tokenize("¿Mitigación del PF-5030 en pagos-api? 503!") == [
        "mitigacion",
        "del",
        "pf-5030",
        "en",
        "pagos-api",
        "503",
    ]


def test_lexical_retrieval_matches_without_accents(retrieval: HybridRetrieval) -> None:
    docs = retrieval.lexical().invoke("mitigacion")

    assert sources(docs)[0] == "runbook/base.md"


def test_vector_retrieval_spans_every_namespace(retrieval: HybridRetrieval) -> None:
    docs = retrieval.vector().invoke("PoolTimeout en pagos-api")

    assert set(sources(docs)) == {
        "errores/catalogo.json",
        "runbook/base.md",
        "arquitectura/flujo.md",
        "log/pagos-api.log",
    }
    scores = [doc.metadata["score"] for doc in docs]
    assert scores == sorted(scores, reverse=True)


def test_a_category_keeps_both_retrievals_inside_its_namespace(retrieval: HybridRetrieval) -> None:
    query = "PoolTimeout en pagos-api"

    assert sources(retrieval.vector(Category.LOG).invoke(query)) == ["log/pagos-api.log"]
    assert sources(retrieval.lexical(Category.LOG).invoke(query)) == ["log/pagos-api.log"]
    assert sources(retrieval.hybrid(HybridWeights(), Category.LOG).invoke(query)) == [
        "log/pagos-api.log"
    ]


def test_weights_decide_which_ranking_leads_the_fusion(retrieval: HybridRetrieval) -> None:
    query = "la base no responde PoolTimeout"
    vector_first = sources(retrieval.vector().invoke(query))[0]
    lexical_first = sources(retrieval.lexical().invoke(query))[0]
    assert vector_first != lexical_first

    assert sources(retrieval.hybrid(HybridWeights(vector=1.0)).invoke(query))[0] == vector_first
    assert sources(retrieval.hybrid(HybridWeights(vector=0.0)).invoke(query))[0] == lexical_first
