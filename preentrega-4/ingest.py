import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import tiktoken

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Vector
from pinecone.db_data import Index
from pinecone.db_data.types import VectorMetadataTypedDict
from pydantic import TypeAdapter

from config import DATA_DIR, chunking_settings, connect, run
from schemas import Category, Chunk, ChunkingSettings, ChunkMetadata, ErrorCode, id_prefix, source_of

SUPPORTED_SUFFIXES = {".md", ".txt", ".log", ".json"}

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

_CATALOG = TypeAdapter(list[ErrorCode])

_ENCODING = "cl100k_base"

_VISIBILITY_TIMEOUT = 120.0


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


def _catalog_chunks(path: Path, source: str, category: Category) -> list[Chunk]:
    error_codes = _CATALOG.validate_json(path.read_bytes())
    return [
        Chunk(
            text=error_code.text,
            metadata=ChunkMetadata(
                source=source, category=category, section=error_code.codigo, chunk_index=index
            ),
        )
        for index, error_code in enumerate(error_codes)
    ]


def load_chunks(path: Path, data_dir: Path, settings: ChunkingSettings) -> list[Chunk]:
    relative = path.relative_to(data_dir)
    category = _category_of(relative)
    if path.suffix == ".json":
        return _catalog_chunks(path, relative.as_posix(), category)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=_ENCODING,
        chunk_size=settings.chunk_tokens,
        chunk_overlap=settings.overlap_tokens,
    )

    chunks: list[Chunk] = []
    for section, text in _sections(path, path.read_text(encoding="utf-8")):
        for piece in splitter.split_text(text):
            metadata = ChunkMetadata(
                source=relative.as_posix(),
                category=category,
                section=section,
                chunk_index=len(chunks),
            )
            chunks.append(Chunk(text=piece, metadata=metadata))
    return chunks


def merge_sections(chunks: Sequence[Chunk], settings: ChunkingSettings) -> list[Chunk]:
    encoding = tiktoken.get_encoding(_ENCODING)
    groups: list[tuple[ChunkMetadata, str]] = []
    for chunk in chunks:
        if groups and groups[-1][0].source == chunk.metadata.source:
            joined = f"{groups[-1][1]}\n\n{chunk.text}"
            if len(encoding.encode(joined)) <= settings.chunk_tokens:
                groups[-1] = (groups[-1][0], joined)
                continue
        groups.append((chunk.metadata, chunk.text))

    positions: Counter[str] = Counter()
    merged = []
    for metadata, text in groups:
        index = positions[metadata.source]
        positions[metadata.source] += 1
        merged.append(Chunk(text=text, metadata=metadata.model_copy(update={"chunk_index": index})))
    return merged


def source_paths(data_dir: Path) -> list[Path]:
    return sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix in SUPPORTED_SUFFIXES
    )


def load_directory(data_dir: Path, settings: ChunkingSettings, *, merged: bool = False) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in source_paths(data_dir):
        sections = load_chunks(path, data_dir, settings)
        # El catálogo nunca se fusiona: cada código de error es un chunk.
        chunks.extend(merge_sections(sections, settings) if merged and path.suffix != ".json" else sections)
    return chunks


def summary(chunks: Sequence[Chunk]) -> str:
    counts = Counter(chunk.metadata.category for chunk in chunks)
    by_namespace = ", ".join(f"{category.value}={counts[category]}" for category in Category)
    return f"{len(chunks)} chunks ({by_namespace})"


def _vector_of(chunk: Chunk, values: list[float]) -> Vector:
    meta = chunk.metadata
    metadata: VectorMetadataTypedDict = {
        "text": chunk.text,
        "source": meta.source,
        "category": meta.category.value,
        "section": meta.section,
        "chunk_index": meta.chunk_index,
    }
    return Vector(id=chunk.id, values=values, metadata=metadata)


class Ingestor:
    def __init__(
        self,
        index: Index,
        embeddings: Embeddings,
        settings: ChunkingSettings,
        data_dir: Path,
    ) -> None:
        self._index = index
        self._embeddings = embeddings
        self._settings = settings
        self._data_dir = data_dir

    def ingest_source(self, path: Path) -> list[Chunk]:
        chunks = load_chunks(path, self._data_dir, self._settings)
        vectors = self._embeddings.embed_documents([chunk.text for chunk in chunks])
        relative = path.relative_to(self._data_dir)
        namespace = _category_of(relative).value
        if chunks:
            self._index.upsert(
                vectors=[_vector_of(chunk, values) for chunk, values in zip(chunks, vectors, strict=True)],
                namespace=namespace,
                batch_size=100,
                show_progress=False,
            )
        # Primero se pisa lo que existe y después se borra lo que sobra: la Source nunca queda sin vectores.
        kept = {chunk.id for chunk in chunks}
        orphans = [i for i in self._ids(namespace, id_prefix(relative.as_posix())) if i not in kept]
        if orphans:
            self._index.delete(ids=orphans, namespace=namespace)
        return chunks

    def ingest_directory(self) -> list[Chunk]:
        chunks = [chunk for path in source_paths(self._data_dir) for chunk in self.ingest_source(path)]
        self._prune({chunk.metadata.source for chunk in chunks})
        self._wait_until_visible(chunks)
        return chunks

    def _ids(self, namespace: str, prefix: str | None = None) -> list[str]:
        return [vector_id for page in self._index.list(namespace=namespace, prefix=prefix) for vector_id in page]

    def _prune(self, sources: set[str]) -> None:
        for category in Category:
            stale = [i for i in self._ids(category.value) if source_of(i) not in sources]
            if stale:
                self._index.delete(ids=stale, namespace=category.value)

    def _wait_until_visible(self, chunks: Sequence[Chunk]) -> None:
        # Pinecone Serverless es eventualmente consistente: contar vectores no alcanza, porque una
        # re-ingesta deja la misma cantidad. Se espera a leer exactamente los IDs y textos subidos.
        expected: dict[str, dict[str, str]] = {category.value: {} for category in Category}
        for chunk in chunks:
            expected[chunk.metadata.category.value][chunk.id] = chunk.text
        deadline = time.monotonic() + _VISIBILITY_TIMEOUT
        while not all(self._shows(namespace, texts) for namespace, texts in expected.items()):
            if time.monotonic() > deadline:
                raise TimeoutError("Pinecone todavía no muestra los vectores subidos.")
            time.sleep(2)

    def _shows(self, namespace: str, texts: dict[str, str]) -> bool:
        if set(self._ids(namespace)) != texts.keys():
            return False
        ids = list(texts)
        for start in range(0, len(ids), 100):
            vectors = self._index.fetch(ids=ids[start : start + 100], namespace=namespace).vectors
            for vector_id in ids[start : start + 100]:
                vector = vectors.get(vector_id)
                if vector is None or (vector.metadata or {}).get("text") != texts[vector_id]:
                    return False
        return True


def main() -> None:
    index, embeddings = connect()
    chunks = Ingestor(index, embeddings, chunking_settings(), DATA_DIR).ingest_directory()
    print(f"Ingestados {summary(chunks)} de {DATA_DIR.name}/ en Pinecone")


if __name__ == "__main__":
    run(main)
