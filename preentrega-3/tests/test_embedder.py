import math
from dataclasses import dataclass, field
from typing import Any

import pytest
from google.genai import errors, types

from embedder import GeminiEmbedder
from schemas import EmbeddingSpace


@dataclass
class FakeModels:
    replies: list[Any] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def embed_content(
        self, *, model: str, contents: list[str], config: types.EmbedContentConfig
    ) -> types.EmbedContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
        return types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[3.0, 4.0]) for _ in contents]
        )


@dataclass
class FakeClient:
    models: FakeModels = field(default_factory=FakeModels)


SPACE = EmbeddingSpace(model="gemini-embedding-001", dimension=2)


def embedder(client: FakeClient, **kwargs: Any) -> GeminiEmbedder:
    return GeminiEmbedder(client, SPACE, backoff=0, **kwargs)  # type: ignore[arg-type]


def test_vectors_come_back_normalized() -> None:
    [vector] = embedder(FakeClient()).embed_documents(["hola"])

    assert vector == pytest.approx([0.6, 0.8])


def test_documents_and_queries_use_different_task_types() -> None:
    client = FakeClient()

    embedder(client).embed_documents(["un chunk"])
    embedder(client).embed_query("una pregunta")

    doc_config, query_config = (call["config"] for call in client.models.calls)
    assert doc_config.task_type == "RETRIEVAL_DOCUMENT"
    assert query_config.task_type == "RETRIEVAL_QUERY"
    assert doc_config.output_dimensionality == 2
    assert client.models.calls[0]["model"] == "gemini-embedding-001"


def test_documents_are_sent_in_batches() -> None:
    client = FakeClient()

    vectors = embedder(client, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert [call["contents"] for call in client.models.calls] == [["a", "b"], ["c", "d"], ["e"]]


def test_rate_limit_is_retried() -> None:
    client = FakeClient(FakeModels(replies=[errors.ClientError(429, {"error": {"message": "quota"}})]))

    [vector] = embedder(client).embed_documents(["hola"])

    assert math.isclose(sum(x * x for x in vector), 1.0)
    assert len(client.models.calls) == 2


def test_other_client_errors_are_not_retried() -> None:
    client = FakeClient(FakeModels(replies=[errors.ClientError(400, {"error": {"message": "bad"}})]))

    with pytest.raises(errors.ClientError):
        embedder(client).embed_documents(["hola"])
    assert len(client.models.calls) == 1


def test_no_texts_means_no_request() -> None:
    client = FakeClient()

    assert embedder(client).embed_documents([]) == []
    assert client.models.calls == []
