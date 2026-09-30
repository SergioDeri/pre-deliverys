import re
import unicodedata
from collections.abc import Mapping, Sequence

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone.db_data import Index
from pydantic import ConfigDict

from schemas import Category, Chunk, HybridWeights

_TOKEN = re.compile(r"\w+(?:-\w+)*")


def tokenize(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    return _TOKEN.findall(plain)


class NamespacedVectorRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stores: list[VectorStore]
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        scored = [
            (doc, score)
            for store in self.stores
            for doc, score in store.similarity_search_with_score(query, k=self.k)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            Document(id=doc.id, page_content=doc.page_content, metadata={**doc.metadata, "score": score})
            for doc, score in scored[: self.k]
        ]


class HybridRetrieval:
    def __init__(
        self,
        stores: Mapping[Category, VectorStore],
        chunks: Sequence[Document],
        *,
        fetch_k: int = 20,
    ) -> None:
        self._stores = dict(stores)
        self._chunks = list(chunks)
        self._fetch_k = fetch_k
        self._lexical: dict[Category | None, BM25Retriever] = {}

    def sources(self) -> set[str]:
        return {str(doc.metadata["source"]) for doc in self._chunks}

    def vector(self, category: Category | None = None) -> NamespacedVectorRetriever:
        if category is None:
            stores = list(self._stores.values())
        else:
            stores = [self._stores[category]]
        return NamespacedVectorRetriever(stores=stores, k=self._fetch_k)

    def lexical(self, category: Category | None = None) -> BM25Retriever:
        if category not in self._lexical:
            chunks = [
                doc
                for doc in self._chunks
                if category is None or doc.metadata["category"] == category
            ]
            if not chunks:
                scope = category or "ningún namespace"
                raise ValueError(f"No hay chunks indexados en {scope}: corré la ingesta primero.")
            self._lexical[category] = BM25Retriever.from_documents(
                chunks, preprocess_func=tokenize, k=self._fetch_k
            )
        return self._lexical[category]

    def hybrid(self, weights: HybridWeights, category: Category | None = None) -> EnsembleRetriever:
        return EnsembleRetriever(
            retrievers=[self.vector(category), self.lexical(category)],
            weights=[weights.vector, weights.lexical],
        )


def pinecone_stores(index: Index, embeddings: Embeddings) -> dict[Category, VectorStore]:
    return {
        category: PineconeVectorStore(index=index, embedding=embeddings, namespace=category.value)
        for category in Category
    }


def load_indexed_chunks(index: Index) -> list[Document]:
    chunks = []
    for category in Category:
        for ids in index.list(namespace=category.value):
            vectors = index.fetch(ids=list(ids), namespace=category.value).vectors
            for vector_id, vector in sorted(vectors.items()):
                metadata = dict(vector.metadata or {})
                text = str(metadata.pop("text"))
                chunks.append(Document(id=vector_id, page_content=text, metadata=metadata))
    return chunks


def pinecone_retrieval(index: Index, embeddings: Embeddings) -> HybridRetrieval:
    return HybridRetrieval(pinecone_stores(index, embeddings), load_indexed_chunks(index))


def in_memory_retrieval(chunks: Sequence[Chunk], embeddings: Embeddings) -> HybridRetrieval:
    documents = [
        Document(id=chunk.id, page_content=chunk.text, metadata=chunk.metadata.model_dump(mode="json"))
        for chunk in chunks
    ]
    stores: dict[Category, VectorStore] = {}
    for category in Category:
        store = InMemoryVectorStore(embeddings)
        in_namespace = [doc for doc in documents if doc.metadata["category"] == category]
        if in_namespace:
            store.add_documents(in_namespace)
        stores[category] = store
    return HybridRetrieval(stores, documents)
