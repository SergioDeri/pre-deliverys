import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from llm_client import ChatMessage, LLMStreamError, ModelResponse, create_client

CONVERSATION = [
    ChatMessage(role="system", content="Respondes en español, en no más de tres frases."),
    ChatMessage(role="user", content="¿Qué ventaja tiene usar asyncio al llamar a una API de LLM?"),
]


def show(response: ModelResponse) -> None:
    if response.error:
        error = response.error
        print(f"  error [{error.kind}] (reintentable: {error.retryable}): {error.message}")
    else:
        print(f"  {response.content}")
    details = [f"latencia: {response.latency_ms:.0f} ms", f"intentos: {response.attempts}"]
    if response.usage:
        details.insert(0, f"tokens: {response.usage.total_tokens}")
    print("  " + " · ".join(details))


async def demo_generate() -> None:
    print("\n== Respuesta completa ==")
    async with create_client() as client:
        show(await client.generate(CONVERSATION))


async def demo_stream() -> None:
    print("\n== Streaming ==")
    async with create_client() as client:
        try:
            async for fragment in client.stream(CONVERSATION):
                print(fragment, end="", flush=True)
            print()
        except LLMStreamError as exc:
            print(f"\n  el stream se cortó [{exc.error.kind}]: {exc.error.message}")


async def demo_invalid_key() -> None:
    print("\n== API key inválida (error controlado) ==")
    async with create_client(api_key="gsk_clave_invalida") as client:
        show(await client.generate(CONVERSATION))


async def main() -> None:
    await demo_generate()
    await demo_stream()
    await demo_invalid_key()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Falta GROQ_API_KEY: copia .env.example a .env y pon tu clave.")
    asyncio.run(main())
