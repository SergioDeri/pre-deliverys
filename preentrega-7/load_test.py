import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

QUESTIONS = (
    "¿Qué pasó el 26/09/2026 en la conciliación y cuántas transacciones quedaron con diferencias?",
    "¿Qué pasó el 27/09/2026 a las 19:41 con Cobralia y cuántos errores hubo por servicio?",
    "¿Cuántos errores hubo por servicio el 28/09/2026 entre las 21:14 y las 21:15?",
    "¿Qué causó los 502 del 29/09/2026 a las 10:03 y cuántos errores hubo por servicio?",
    "¿Cuánto duró el incidente del 28/09/2026 a las 21:14 y qué servicio falló primero?",
)
FINISHED = ("DONE", "FAILED")
POLL_SECONDS = 2.0


def p95(values: list[float]) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=20, method="inclusive")[18]


async def run_one(client: httpx.AsyncClient, question: str) -> dict[str, Any]:
    sent = time.perf_counter()
    response = await client.post("/tasks", json={"question": question})
    response.raise_for_status()
    accepted_ms = (time.perf_counter() - sent) * 1000
    job_id = response.json()["id"]
    while True:
        job: dict[str, Any] = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] == "PAUSED_FOR_APPROVAL":
            approval = {"approved": True, "approver": "prueba-de-carga", "comment": "aprobación automática"}
            await client.post(f"/tasks/{job_id}/approve", json=approval)
        elif job["status"] in FINISHED:
            break
        await asyncio.sleep(POLL_SECONDS)
    return {
        "id": job_id,
        "question": question,
        "status": job["status"],
        "accepted_ms": accepted_ms,
        "end_to_end_s": time.perf_counter() - sent,
        "run_seconds": job["run_seconds"],
        "input_tokens": job["usage"]["input_tokens"],
        "output_tokens": job["usage"]["output_tokens"],
        "cost_usd": job["usage"]["cost_usd"],
        "error": job["error"],
    }


def report(results: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in results if r["status"] == "DONE"]
    return {
        "jobs": len(results),
        "done": len(done),
        "failed": len(results) - len(done),
        "p95_accepted_ms": p95([r["accepted_ms"] for r in results]),
        "p95_run_seconds": p95([r["run_seconds"] for r in done]),
        "p95_end_to_end_s": p95([r["end_to_end_s"] for r in done]),
        "mean_cost_usd": statistics.fmean(r["cost_usd"] for r in done) if done else 0.0,
        "total_cost_usd": sum(r["cost_usd"] for r in results),
        "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in results),
    }


async def main(base_url: str, output: Path) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        results = await asyncio.gather(*(run_one(client, question) for question in QUESTIONS))
    for r in results:
        print(
            f"{r['status']:<6} {r['accepted_ms']:6.1f} ms  ejecución {r['run_seconds']:6.1f} s  "
            f"{r['input_tokens'] + r['output_tokens']:6d} tokens  US$ {r['cost_usd']:.5f}  "
            f"{r['question'][:60]}"
        )
        if r["error"]:
            print(f"       {r['error']}")
    summary = report(results)
    print(
        f"\n{summary['done']}/{summary['jobs']} terminados | p95 POST /tasks: {summary['p95_accepted_ms']:.1f} ms | "
        f"p95 ejecución: {summary['p95_run_seconds']:.1f} s | p95 punta a punta: {summary['p95_end_to_end_s']:.1f} s | "
        f"costo medio: US$ {summary['mean_cost_usd']:.5f} | total: US$ {summary['total_cost_usd']:.5f}"
    )
    output.write_text(json.dumps({"summary": summary, "jobs": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados en {output.name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cinco investigaciones simultáneas contra la API")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    asyncio.run(main(args.url, Path(__file__).parent / "load_test_results.json"))
