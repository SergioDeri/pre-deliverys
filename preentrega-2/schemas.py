from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


class InputKind(StrEnum):
    ARQUITECTURA = "arquitectura"
    LOG_ERROR = "log_error"
    AMBIGUO = "ambiguo"


class CriticalityLevel(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


Summary = Annotated[str, StringConstraints(strip_whitespace=True, min_length=20)]

_PLACEHOLDERS = {
    "n/a", "na", "-", "ninguna", "ninguno", "desconocida", "desconocido", "none", "unknown"
}


def _clean_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned = []
    for name in names:
        name = name.strip()
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            cleaned.append(name)
    return cleaned


class TechnicalAnalysis(BaseModel):
    """Análisis técnico extraído de una descripción de arquitectura o de un log de error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tipo_de_entrada: InputKind = Field(
        description="Qué es el texto: arquitectura, log_error o ambiguo."
    )
    tecnologias: list[str] = Field(
        description="Herramientas, lenguajes, frameworks o servicios nombrados en el texto."
    )
    componentes_afectados: list[str] = Field(
        default_factory=list,
        description="Partes del sistema implicadas o que fallan según el texto.",
    )
    nivel_de_criticidad: CriticalityLevel = Field(
        description="Urgencia de la situación descrita: baja, media o alta."
    )
    resumen_tecnico: Summary = Field(description="Resumen breve pensado para un ingeniero.")

    @field_validator("tecnologias")
    @classmethod
    def _at_least_one_technology(cls, value: list[str]) -> list[str]:
        cleaned = [name for name in _clean_names(value) if name.casefold() not in _PLACEHOLDERS]
        if not cleaned:
            raise ValueError("debe nombrar al menos una tecnología concreta")
        return cleaned

    @field_validator("componentes_afectados")
    @classmethod
    def _clean_components(cls, value: list[str]) -> list[str]:
        return _clean_names(value)

    @model_validator(mode="after")
    def _consistent_with_input_kind(self) -> Self:
        if self.tipo_de_entrada is InputKind.LOG_ERROR and not self.componentes_afectados:
            raise ValueError("un log de error debe indicar al menos un componente afectado")
        if (
            self.tipo_de_entrada is InputKind.AMBIGUO
            and self.nivel_de_criticidad is CriticalityLevel.ALTA
        ):
            raise ValueError("un texto ambiguo no justifica una criticidad alta")
        return self


FailureKind = Literal["invalid_output", "truncated_output", "provider"]


class ExtractionFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: FailureKind
    message: str


class ExtractionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis: TechnicalAnalysis | None = None
    failure: ExtractionFailure | None = None
    latency_ms: float = 0.0

    @model_validator(mode="after")
    def _either_analysis_or_failure(self) -> Self:
        if (self.analysis is None) == (self.failure is None):
            raise ValueError("debe tener exactamente uno de analysis o failure")
        return self

    @property
    def ok(self) -> bool:
        return self.analysis is not None
