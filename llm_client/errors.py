from .schemas import LLMError


class LLMStreamError(Exception):
    def __init__(self, error: LLMError) -> None:
        super().__init__(f"{error.provider} {error.kind}: {error.message}")
        self.error = error
