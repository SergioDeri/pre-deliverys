from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class Category(StrEnum):
    ARQUITECTURA = "arquitectura"
    RUNBOOK = "runbook"
    LOG = "log"
    ERRORES = "errores"


class EmbeddingSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    dimension: int = Field(gt=0)


class ChunkingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_tokens: int = Field(default=600, gt=0)
    overlap_tokens: int = Field(default=80, ge=0)

    @model_validator(mode="after")
    def _overlap_below_size(self) -> Self:
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens tiene que ser menor que chunk_tokens")
        return self


def chunk_id(source: str, index: int) -> str:
    return f"{id_prefix(source)}{index}"


def id_prefix(source: str) -> str:
    return f"{source}#"


def source_of(vector_id: str) -> str:
    return vector_id.rsplit("#", 1)[0]


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    category: Category
    section: str = ""
    chunk_index: int = Field(ge=0)


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    metadata: ChunkMetadata

    @property
    def id(self) -> str:
        return chunk_id(self.metadata.source, self.metadata.chunk_index)


class ErrorCode(BaseModel):
    model_config = ConfigDict(frozen=True)

    codigo: str
    titulo: str
    http: int | None = None
    causa: str
    runbook: str | None = None

    @property
    def text(self) -> str:
        status = f" (HTTP {self.http})" if self.http else ""
        lines = [f"{self.codigo}: {self.titulo}{status}", f"Causa: {self.causa}"]
        if self.runbook:
            lines.append(f"Runbook: {self.runbook}")
        return "\n".join(lines)


class HybridWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    vector: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def lexical(self) -> float:
        return round(1.0 - self.vector, 6)

    @property
    def label(self) -> str:
        return f"híbrido {self.vector:.1f}/{self.lexical:.1f}"


Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class GoldenCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    question: Question = Field(alias="pregunta")
    expected_sources: frozenset[str] = Field(alias="documentos_esperados", min_length=1)


class RetrievalScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    k: int
    precision: float
    recall: float
