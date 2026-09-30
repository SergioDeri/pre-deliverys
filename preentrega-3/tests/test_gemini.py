import math
import os

import pytest
from dotenv import load_dotenv
from google import genai

from embedder import GeminiEmbedder
from schemas import EmbeddingSpace

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="hace falta GEMINI_API_KEY para llamar a Gemini"
)


def test_real_query_lands_closer_to_the_relevant_chunk() -> None:
    embedder = GeminiEmbedder(
        genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
        EmbeddingSpace(model="gemini-embedding-001", dimension=768),
    )
    database, rollback = embedder.embed_documents(
        [
            "Si PostgreSQL no responde, el servicio de pagos agota el pool de conexiones.",
            "Para volver atrás un deploy se sincroniza la revisión anterior en Argo CD.",
        ]
    )
    query = embedder.embed_query("¿qué pasa cuando la base de datos no contesta?")

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    assert len(query) == 768
    assert math.isclose(sum(x * x for x in database), 1.0, rel_tol=1e-6)
    assert cosine(query, database) > cosine(query, rollback)
