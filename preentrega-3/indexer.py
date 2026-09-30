from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from embedder import Embedder
from schemas import Category, Chunk, ChunkingSettings, ChunkMetadata, EmbeddingSpace

SUPPORTED_SUFFIXES = {".md", ".txt", ".log"}

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _category_of(relative: Path) -> Category:
    folder = relative.parts[0] if len(relative.parts) > 1 else ""
    try:
        return Category(folder)
    except ValueError:
        valid = ", ".join(category.value for category in Category)
        raise ValueError(
            f"{relative}: la carpeta '{folder}' no es una categoría válida ({valid})"
        ) from None


def _sections(path: Path, text: str) -> list[tuple[str, str]]:
    if path.suffix != ".md":
        return [("", text)]
    splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
    sections = []
    for doc in splitter.split_text(text):
        headers = [doc.metadata[key] for _, key in _HEADERS if key in doc.metadata]
        sections.append((headers[-1] if headers else "", doc.page_content))
    return sections


def load_chunks(
    path: Path, data_dir: Path, settings: ChunkingSettings, now: datetime | None = None
) -> list[Chunk]:
    relative = path.relative_to(data_dir)
    category = _category_of(relative)
    ingested_at = now or datetime.now(UTC)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    chunks: list[Chunk] = []
    for section, text in _sections(path, path.read_text(encoding="utf-8")):
        for piece in splitter.split_text(text):
            metadata = ChunkMetadata(
                source=relative.as_posix(),
                category=category,
                section=section,
                chunk_index=len(chunks),
                ingested_at=ingested_at,
            )
            chunks.append(Chunk(text=piece, metadata=metadata))
    return chunks


class EmbeddingSpaceMismatch(Exception):
    pass


def open_collection(client: ClientAPI, name: str, space: EmbeddingSpace) -> Collection:
    try:
        collection = client.get_collection(name, embedding_function=None)
    except NotFoundError:
        return client.create_collection(
            name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": space.model, "embedding_dimension": space.dimension},
        )

    metadata = collection.metadata or {}
    stored = EmbeddingSpace(
        model=str(metadata.get("embedding_model")),
        dimension=int(metadata.get("embedding_dimension", 0)),
    )
    if stored != space:
        raise EmbeddingSpaceMismatch(
            f"La colección '{name}' se creó con {stored.model} ({stored.dimension} dims) "
            f"y ahora se pide {space.model} ({space.dimension} dims). "
            "Borra la carpeta de ChromaDB o usa otra colección."
        )
    return collection


class Indexer:
    def __init__(
        self,
        collection: Collection,
        embedder: Embedder,
        settings: ChunkingSettings,
        data_dir: Path,
    ) -> None:
        self._collection = collection
        self._embedder = embedder
        self._settings = settings
        self._data_dir = data_dir

    def index_source(self, path: Path) -> int:
        chunks = load_chunks(path, self._data_dir, self._settings)
        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])
        # Borrar antes del upsert evita chunks huérfanos si el archivo ahora es más corto.
        self.delete_source(path.relative_to(self._data_dir).as_posix())
        if chunks:
            self._collection.upsert(
                ids=[chunk.id for chunk in chunks],
                embeddings=np.array(embeddings, dtype=np.float32),
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata.model_dump(mode="json") for chunk in chunks],
            )
        return len(chunks)

    def index_directory(self) -> int:
        paths = sorted(
            path
            for path in self._data_dir.rglob("*")
            if path.is_file() and path.suffix in SUPPORTED_SUFFIXES
        )
        return sum(self.index_source(path) for path in paths)

    def delete_source(self, source: str) -> int:
        ids = self._collection.get(where={"source": source}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self._collection.count()
