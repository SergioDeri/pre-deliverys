import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from phoenix.otel import register

DEFAULT_PROJECT = "incident-orchestrator"


def setup_tracing() -> TracerProvider | None:
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return None
    provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME") or DEFAULT_PROJECT,
        endpoint=endpoint,
        batch=True,
        set_global_tracer_provider=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return provider
