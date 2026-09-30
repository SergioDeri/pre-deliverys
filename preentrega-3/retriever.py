import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Where

from embedder import Embedder
from schemas import ChunkMetadata, SearchQuery, SearchResult


def _metadata_filter(query: SearchQuery) -> Where | None:
    conditions: list[Where] = []
    if query.category is not None:
        conditions.append({"category": query.category.value})
    if query.source is not None:
        conditions.append({"source": query.source})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class Retriever:
    def __init__(self, collection: Collection, embedder: Embedder) -> None:
        self._collection = collection
        self._embedder = embedder

    def search(self, query: SearchQuery) -> list[SearchResult]:
        vector = self._embedder.embed_query(query.text)
        response = self._collection.query(
            query_embeddings=np.array([vector], dtype=np.float32),
            n_results=query.k,
            where=_metadata_filter(query),
            include=["documents", "metadatas", "distances"],
        )
        documents = (response["documents"] or [[]])[0]
        metadatas = (response["metadatas"] or [[]])[0]
        distances = (response["distances"] or [[]])[0]

        results = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=True):
            # Distancia coseno de Chroma: 0 es idéntico y 2 es opuesto.
            score = 1.0 - distance
            if query.min_score is not None and score < query.min_score:
                continue
            results.append(
                SearchResult(
                    text=text,
                    score=score,
                    distance=distance,
                    metadata=ChunkMetadata.model_validate(metadata),
                )
            )
        return results
