import hashlib
import math
import re
from collections.abc import Sequence

from schemas import EmbeddingSpace


class FakeEmbedder:
    """Bolsa de palabras hasheada: textos con palabras en común quedan cerca."""

    def __init__(self, dimension: int = 64, model: str = "fake-embedding") -> None:
        self.space = EmbeddingSpace(model=model, dimension=dimension)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.space.dimension
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.space.dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
