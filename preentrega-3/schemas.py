from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class Category(StrEnum):
    ARQUITECTURA = "arquitectura"
    RUNBOOK = "runbook"
    LOG = "log"


class EmbeddingSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    dimension: int = Field(gt=0)


class ChunkingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def _overlap_below_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap tiene que ser menor que chunk_size")
        return self


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    category: Category
    section: str = ""
    chunk_index: int = Field(ge=0)
    ingested_at: datetime


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    metadata: ChunkMetadata

    @property
    def id(self) -> str:
        return f"{self.metadata.source}#{self.metadata.chunk_index}"


QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: QueryText
    k: int = Field(default=4, ge=1, le=50)
    category: Category | None = None
    source: str | None = None
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    score: float
    distance: float
    metadata: ChunkMetadata
