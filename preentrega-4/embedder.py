import math
from collections.abc import Sequence

from google import genai
from google.genai import errors, types
from langchain_core.embeddings import Embeddings
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from schemas import EmbeddingSpace


def _is_transient(error: BaseException) -> bool:
    if isinstance(error, errors.ServerError):
        return True
    return isinstance(error, errors.ClientError) and error.code == 429


def _normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm else list(values)


class GeminiEmbeddings(Embeddings):
    def __init__(
        self,
        client: genai.Client,
        space: EmbeddingSpace,
        *,
        batch_size: int = 100,
        max_attempts: int = 5,
        backoff: float = 2.0,
    ) -> None:
        self.space = space
        self._client = client
        self._batch_size = batch_size
        self._queries: dict[str, list[float]] = {}
        self._retrying = Retrying(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=backoff, max=60),
            reraise=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed(texts[start : start + self._batch_size], "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        # El retriever vectorial consulta un namespace por vez con la misma pregunta.
        if text not in self._queries:
            self._queries[text] = self._embed([text], "RETRIEVAL_QUERY")[0]
        return self._queries[text]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.space.dimension
        )
        response = self._retrying(
            self._client.models.embed_content,
            model=self.space.model,
            contents=texts,
            config=config,
        )
        # Por debajo de 3072 dimensiones Gemini no normaliza, y el índice usa coseno.
        return [_normalize(embedding.values or []) for embedding in response.embeddings or []]
