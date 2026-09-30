from typing import Any

import pytest
from pydantic import ValidationError

from schemas import (
    CriticalityLevel,
    ExtractionFailure,
    ExtractionOutcome,
    InputKind,
    TechnicalAnalysis,
)


def analysis(**overrides: Any) -> TechnicalAnalysis:
    fields: dict[str, Any] = {
        "tipo_de_entrada": "log_error",
        "tecnologias": ["PostgreSQL", "FastAPI"],
        "componentes_afectados": ["api-pedidos"],
        "nivel_de_criticidad": "alta",
        "resumen_tecnico": "El pool de conexiones a PostgreSQL se agota y la API responde 503.",
    }
    return TechnicalAnalysis(**(fields | overrides))


def test_valid_log_analysis_keeps_its_values() -> None:
    result = analysis()

    assert result.tipo_de_entrada is InputKind.LOG_ERROR
    assert result.nivel_de_criticidad is CriticalityLevel.ALTA
    assert result.tecnologias == ["PostgreSQL", "FastAPI"]


def test_technologies_are_trimmed_and_deduplicated_ignoring_case() -> None:
    result = analysis(tecnologias=["  Redis ", "redis", "", "Kafka"])

    assert result.tecnologias == ["Redis", "Kafka"]


@pytest.mark.parametrize("technologies", [[], ["", "   "]])
def test_analysis_needs_at_least_one_technology(technologies: list[str]) -> None:
    with pytest.raises(ValidationError, match="tecnolog"):
        analysis(tecnologias=technologies)


@pytest.mark.parametrize("placeholder", ["N/A", "desconocida", "Ninguna", "unknown", "-"])
def test_placeholders_do_not_count_as_technologies(placeholder: str) -> None:
    with pytest.raises(ValidationError, match="tecnolog"):
        analysis(tecnologias=[placeholder])


def test_affected_components_are_cleaned_and_may_be_empty() -> None:
    result = analysis(tipo_de_entrada="arquitectura", componentes_afectados=[" web ", "WEB", " "])
    empty = analysis(tipo_de_entrada="arquitectura", componentes_afectados=[])

    assert result.componentes_afectados == ["web"]
    assert empty.componentes_afectados == []


@pytest.mark.parametrize("summary", ["", "      ", "Falla la BD."])
def test_summary_must_be_descriptive(summary: str) -> None:
    with pytest.raises(ValidationError, match="resumen_tecnico"):
        analysis(resumen_tecnico=summary)


def test_summary_is_trimmed() -> None:
    result = analysis(resumen_tecnico="  Redis actúa como caché de sesiones del frontend.\n")

    assert result.resumen_tecnico == "Redis actúa como caché de sesiones del frontend."


def test_error_log_must_name_an_affected_component() -> None:
    with pytest.raises(ValidationError, match="componente"):
        analysis(tipo_de_entrada="log_error", componentes_afectados=[" "])


def test_ambiguous_text_cannot_be_highly_critical() -> None:
    with pytest.raises(ValidationError, match="ambiguo"):
        analysis(tipo_de_entrada="ambiguo", nivel_de_criticidad="alta")


def test_ambiguous_text_with_lower_criticality_is_valid() -> None:
    result = analysis(tipo_de_entrada="ambiguo", nivel_de_criticidad="media")

    assert result.nivel_de_criticidad is CriticalityLevel.MEDIA


@pytest.mark.parametrize(
    "overrides",
    [
        {"nivel_de_criticidad": "critica"},
        {"tipo_de_entrada": "metricas"},
        {"prioridad": "p1"},
    ],
)
def test_analysis_rejects_values_outside_the_schema(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        analysis(**overrides)


def test_outcome_with_an_analysis_is_ok() -> None:
    outcome = ExtractionOutcome(analysis=analysis())

    assert outcome.ok


def test_outcome_with_a_failure_is_not_ok() -> None:
    failure = ExtractionFailure(kind="provider", message="HTTP 401: Invalid API Key")

    assert not ExtractionOutcome(failure=failure).ok


def test_outcome_cannot_hold_both_analysis_and_failure() -> None:
    failure = ExtractionFailure(kind="provider", message="HTTP 401: Invalid API Key")

    with pytest.raises(ValidationError, match="exactamente uno"):
        ExtractionOutcome(analysis=analysis(), failure=failure)


def test_outcome_cannot_be_empty() -> None:
    with pytest.raises(ValidationError, match="exactamente uno"):
        ExtractionOutcome()
