import logging
import os
import sys
import textwrap
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pydantic import ValidationError

from embedder import GeminiEmbedder
from indexer import EmbeddingSpaceMismatch, Indexer, open_collection
from retriever import Retriever
from schemas import ChunkingSettings, EmbeddingSpace, SearchQuery, SearchResult

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDING_DIMENSION = 768

QUESTION = "¿Por qué pagos-api devuelve 503 cuando se agotan las conexiones a la base?"

TEMP_SOURCE = DATA_DIR / "runbook" / "demo-temporal.md"
TEMP_V1 = """\
# Runbook temporal: certificado TLS vencido

## Síntomas

Los comercios reciben errores de handshake al llamar a la API.

## Pasos

Renovar el certificado en el secreto del api-gateway y recargar NGINX.
"""
TEMP_V2 = "# Runbook temporal: certificado TLS vencido\n\nLo renueva cert-manager solo.\n"


def show(title: str, results: list[SearchResult]) -> None:
    print(f"\n== {title} ==")
    if not results:
        print("  (sin resultados)")
    for result in results:
        meta = result.metadata
        where = meta.source + (f" › {meta.section}" if meta.section else "")
        excerpt = textwrap.shorten(" ".join(result.text.split()), width=110)
        print(f"  {result.score:.3f}  [{meta.category}] {where}")
        print(f"         {excerpt}")


def demo_crud(indexer: Indexer, retriever: Retriever) -> None:
    source = TEMP_SOURCE.relative_to(DATA_DIR).as_posix()
    print(f"\n== CRUD sobre {source} ==")
    try:
        TEMP_SOURCE.write_text(TEMP_V1, encoding="utf-8")
        print(f"  alta:        {indexer.index_source(TEMP_SOURCE)} chunks, total {indexer.count()}")

        TEMP_SOURCE.write_text(TEMP_V2, encoding="utf-8")
        print(f"  reindexado:  {indexer.index_source(TEMP_SOURCE)} chunks, total {indexer.count()}")
        show(
            "Consulta sobre el runbook actualizado",
            retriever.search(SearchQuery(text="certificado TLS", k=3, source=source)),
        )

        print(f"\n  baja:        {indexer.delete_source(source)} chunks borrados, total {indexer.count()}")
    finally:
        indexer.delete_source(source)
        TEMP_SOURCE.unlink(missing_ok=True)


def main() -> None:
    space = EmbeddingSpace(
        model=os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001",
        dimension=EMBEDDING_DIMENSION,
    )
    settings = ChunkingSettings.model_validate(
        {
            field: value
            for field, value in {
                "chunk_size": os.getenv("CHUNK_SIZE"),
                "chunk_overlap": os.getenv("CHUNK_OVERLAP"),
            }.items()
            if value
        }
    )
    chroma_dir = BASE_DIR / (os.getenv("CHROMA_DIR") or "chroma_db")

    embedder = GeminiEmbedder(genai.Client(api_key=os.environ["GEMINI_API_KEY"]), space)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = open_collection(client, os.getenv("CHROMA_COLLECTION") or "docs", space)
    indexer = Indexer(collection, embedder, settings, DATA_DIR)
    retriever = Retriever(collection, embedder)

    stored = indexer.index_directory()
    print(f"Indexados {stored} chunks de {DATA_DIR.name}/ en {chroma_dir.name}/ ({space.model}, {space.dimension} dims)")

    show(f"Consulta directa: {QUESTION}", retriever.search(SearchQuery(text=QUESTION, k=5)))
    show(
        f"Consulta filtrada (category=runbook): {QUESTION}",
        retriever.search(SearchQuery(text=QUESTION, k=5, category="runbook")),
    )
    demo_crud(indexer, retriever)


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("Falta GEMINI_API_KEY: copia .env.example a .env y pon tu clave.")
    try:
        main()
    except EmbeddingSpaceMismatch as error:
        sys.exit(str(error))
    except ValidationError as error:
        sys.exit(f"Configuración inválida en el .env:\n{error}")
    except errors.APIError as error:
        sys.exit(f"Gemini respondió {error.code}: {error.message}")
