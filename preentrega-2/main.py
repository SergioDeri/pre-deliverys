import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from chain import build_chain, create_llm, extract, extract_many
from schemas import ExtractionOutcome

ERROR_LOG = """\
2026-09-28 14:02:11 ERROR [api-pedidos] psycopg_pool.PoolTimeout: couldn't get a connection after 30.00 sec
2026-09-28 14:02:11 ERROR [api-pedidos] POST /orders -> 503 Service Unavailable (FastAPI, uvicorn worker 3)
2026-09-28 14:02:12 WARN  [nginx] upstream api-pedidos timed out, 214 requests returned 503 in the last minute
2026-09-28 14:02:14 ERROR [worker-facturacion] celery task invoices.generate failed: OperationalError (PostgreSQL)"""

ARCHITECTURE = """\
El frontend es una SPA en React servida desde un CDN. Habla con una API REST en Django
que guarda los datos en PostgreSQL y usa Redis como caché de sesiones. Las tareas lentas
(envío de correos y generación de PDFs) se encolan en Celery con RabbitMQ como broker.
Todo corre en un clúster de Kubernetes con despliegues gestionados por Argo CD."""

AMBIGUOUS = "A veces el sistema va lento por las tardes y algunos usuarios se quejan."

EXAMPLES = {
    "Log de error": ERROR_LOG,
    "Descripción de arquitectura": ARCHITECTURE,
    "Caso ambiguo": AMBIGUOUS,
}


def show(title: str, outcome: ExtractionOutcome) -> None:
    print(f"\n== {title} ==")
    if outcome.analysis:
        print(outcome.analysis.model_dump_json(indent=2))
    elif outcome.failure:
        print(f"  extracción fallida [{outcome.failure.kind}]: {outcome.failure.message}")
    print(f"  latencia: {outcome.latency_ms:.0f} ms")


async def demo_examples() -> None:
    chain = build_chain(create_llm())
    outcomes = await extract_many(chain, list(EXAMPLES.values()))
    for title, outcome in zip(EXAMPLES, outcomes):
        show(title, outcome)


async def demo_invalid_key() -> None:
    chain = build_chain(create_llm(api_key="gsk_clave_invalida"))
    show("API key inválida (error controlado)", await extract(chain, ERROR_LOG))


async def main() -> None:
    await demo_examples()
    await demo_invalid_key()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Falta GROQ_API_KEY: copia .env.example a .env y pon tu clave.")
    asyncio.run(main())
