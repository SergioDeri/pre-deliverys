import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pinecone import Pinecone, ServerlessSpec
from pinecone.db_data import Index
from pinecone.exceptions import PineconeException
from pydantic import ValidationError

from embedder import GeminiEmbeddings
from schemas import ChunkingSettings, EmbeddingSpace, HybridWeights

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
GOLDEN_DATASET = BASE_DIR / "golden_dataset.json"
EMBEDDING_DIMENSION = 1536
REQUIRED_KEYS = ("GEMINI_API_KEY", "PINECONE_API_KEY")


class EmbeddingSpaceMismatch(Exception):
    pass


def _settings_from_env(fields: dict[str, str]) -> dict[str, str]:
    return {field: value for field, env in fields.items() if (value := os.getenv(env) or "")}


def embedding_space() -> EmbeddingSpace:
    return EmbeddingSpace(
        model=os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001",
        dimension=EMBEDDING_DIMENSION,
    )


def chunking_settings() -> ChunkingSettings:
    return ChunkingSettings.model_validate(
        _settings_from_env({"chunk_tokens": "CHUNK_TOKENS", "overlap_tokens": "CHUNK_OVERLAP_TOKENS"})
    )


def hybrid_weights() -> HybridWeights:
    return HybridWeights.model_validate(_settings_from_env({"vector": "HYBRID_VECTOR_WEIGHT"}))


def open_index(client: Pinecone, name: str, space: EmbeddingSpace) -> Index:
    if not client.has_index(name):
        client.create_index(
            name,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            dimension=space.dimension,
            metric="cosine",
            tags={"embedding_model": space.model},
        )
        return client.Index(name)

    description = client.describe_index(name)
    tags = description.tags or {}
    stored = EmbeddingSpace(
        model=tags.get("embedding_model", "desconocido"), dimension=description.dimension
    )
    if stored != space or description.metric != "cosine":
        raise EmbeddingSpaceMismatch(
            f"El índice '{name}' se creó con {stored.model} ({stored.dimension} dims, "
            f"{description.metric}) y ahora se pide {space.model} ({space.dimension} dims, cosine). "
            "Borralo desde la consola de Pinecone o usá otro PINECONE_INDEX_NAME."
        )
    return client.Index(name)


def connect() -> tuple[Index, GeminiEmbeddings]:
    space = embedding_space()
    embeddings = GeminiEmbeddings(genai.Client(api_key=os.environ["GEMINI_API_KEY"]), space)
    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = open_index(client, os.getenv("PINECONE_INDEX_NAME") or "payflow", space)
    return index, embeddings


def run(main: Callable[[], None]) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        sys.exit(f"Falta {', '.join(missing)}: copiá .env.example a .env en la raíz y completalo.")
    try:
        main()
    except EmbeddingSpaceMismatch as error:
        sys.exit(str(error))
    except ValidationError as error:
        sys.exit(f"Configuración inválida en el .env:\n{error}")
    except errors.APIError as error:
        sys.exit(f"Gemini respondió {error.code}: {error.message}")
    except PineconeException as error:
        sys.exit(f"Pinecone falló: {error}")
