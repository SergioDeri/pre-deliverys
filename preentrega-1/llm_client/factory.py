import os
from collections.abc import Callable

from .base import BaseLLMClient
from .groq_client import GroqClient
from .schemas import GenerationConfig, Provider, RetryPolicy

DEFAULT_MODELS: dict[Provider, str] = {
    Provider.GROQ: "openai/gpt-oss-20b",
}

_Builder = Callable[[GenerationConfig, str | None, RetryPolicy | None], BaseLLMClient]

_BUILDERS: dict[Provider, _Builder] = {
    Provider.GROQ: lambda config, api_key, retry: GroqClient(
        config, api_key=api_key, retry_policy=retry
    ),
}


def create_client(
    provider: Provider | str | None = None,
    *,
    config: GenerationConfig | None = None,
    api_key: str | None = None,
    retry_policy: RetryPolicy | None = None,
) -> BaseLLMClient:
    """Build the client for ``provider``, defaulting to LLM_PROVIDER from the environment."""
    chosen = Provider(provider or os.getenv("LLM_PROVIDER") or Provider.GROQ)
    if config is None:
        model = os.getenv(f"{chosen.upper()}_MODEL") or DEFAULT_MODELS[chosen]
        config = GenerationConfig(model=model)
    return _BUILDERS[chosen](config, api_key, retry_policy)
