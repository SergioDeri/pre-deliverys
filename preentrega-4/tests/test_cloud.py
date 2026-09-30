import math
import os

import pytest
from dotenv import load_dotenv

from config import EMBEDDING_DIMENSION, connect
from hybrid_retriever import pinecone_retrieval
from schemas import HybridWeights

load_dotenv()

pytestmark = pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("PINECONE_API_KEY")),
    reason="hace falta GEMINI_API_KEY y PINECONE_API_KEY para llamar a Gemini y Pinecone",
)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_gemini_embeds_documents_and_queries_asymmetrically() -> None:
    _, embeddings = connect()
    database, rollback = embeddings.embed_documents(
        [
            "Si PostgreSQL no responde, el servicio de pagos agota el pool de conexiones.",
            "Para volver atrás un deploy se sincroniza la revisión anterior en Argo CD.",
        ]
    )
    query = embeddings.embed_query("¿qué pasa cuando la base de datos no contesta?")

    assert len(query) == EMBEDDING_DIMENSION
    assert math.isclose(sum(x * x for x in database), 1.0, rel_tol=1e-6)
    assert cosine(query, database) > cosine(query, rollback)


def test_hybrid_search_over_the_real_index_finds_an_error_code() -> None:
    index, embeddings = connect()
    if not index.describe_index_stats().total_vector_count:
        pytest.skip("el índice está vacío: corré la ingesta primero")

    docs = pinecone_retrieval(index, embeddings).hybrid(HybridWeights()).invoke("PF-4012")

    assert docs[0].metadata["source"] == "errores/catalogo.json"
    assert docs[0].metadata["section"] == "PF-4012"
