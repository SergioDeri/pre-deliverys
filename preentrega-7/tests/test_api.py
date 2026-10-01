import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fakeredis.aioredis import FakeRedis
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph
from app.hitl import PUBLISHED_KEY, RedisIncidentChannel
from app.jobs import JobStore
from app.main import Services, create_app
from app.worker import JobRunner
from tests.fakes import ScriptedChatModel, Stall, decide

QUESTION = "¿Qué pasó el 28/09/2026 a las 21:15?"


def finished(answer: str = "El pool de pagos-api se agotó.") -> list[AIMessage | Exception | Stall]:
    return [
        decide("researcher", "Logs del 28/09"),
        AIMessage(content="12 líneas."),
        decide("FINISH"),
        AIMessage(content=answer),
    ]


class Api:
    def __init__(self, client: AsyncClient, services: Services, redis: FakeRedis) -> None:
        self.client = client
        self.services = services
        self.redis = redis

    async def submit(self, question: str = QUESTION) -> str:
        response = await self.client.post("/tasks", json={"question": question})
        assert response.status_code == 202
        return str(response.json()["id"])

    async def job(self, job_id: str) -> dict[str, Any]:
        response = await self.client.get(f"/tasks/{job_id}")
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        return body

    async def settle(self) -> None:
        await self.services.runner.join()


@asynccontextmanager
async def running_api(
    script: list[AIMessage | Exception | Stall], max_concurrent: int = 5, redis: FakeRedis | None = None
) -> AsyncIterator[Api]:
    redis = redis or FakeRedis(decode_responses=True)
    jobs = JobStore(redis)
    graph = build_graph(ScriptedChatModel(script=script), InMemorySaver(), RedisIncidentChannel(redis))
    services = Services(jobs=jobs, runner=JobRunner(graph, jobs, max_concurrent))

    @asynccontextmanager
    async def provide() -> AsyncIterator[Services]:
        yield services

    app = create_app(provide)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield Api(client, services, redis)


async def test_new_job_is_accepted_at_once_and_pauses_for_approval() -> None:
    async with running_api(finished()) as api:
        response = await api.client.post("/tasks", json={"question": QUESTION})
        assert response.status_code == 202
        assert response.json()["status"] == "PENDING"

        await api.settle()
        job = await api.job(response.json()["id"])

    assert job["status"] == "PAUSED_FOR_APPROVAL"
    assert job["draft"] == "El pool de pagos-api se agotó."
    assert [step["kind"] for step in job["steps"]] == ["delegation", "contribution"]
    assert job["final_answer"] is None
    assert job["paused_at"] is not None


async def test_approval_publishes_the_draft_and_finishes_the_job() -> None:
    async with running_api(finished()) as api:
        job_id = await api.submit()
        await api.settle()

        response = await api.client.post(
            f"/tasks/{job_id}/approve", json={"approved": True, "approver": "ana", "comment": "ok"}
        )
        assert response.status_code == 202
        await api.settle()
        job = await api.job(job_id)
        channel = await api.redis.lrange(PUBLISHED_KEY, 0, -1)

    assert job["status"] == "DONE"
    assert job["final_answer"] == "El pool de pagos-api se agotó."
    assert job["published"] is True
    assert job["approval"]["approver"] == "ana"
    assert job["draft"] is None
    assert [json.loads(entry)["report"] for entry in channel] == ["El pool de pagos-api se agotó."]


async def test_rejection_finishes_the_job_without_publishing() -> None:
    async with running_api(finished()) as api:
        job_id = await api.submit()
        await api.settle()
        await api.client.post(f"/tasks/{job_id}/approve", json={"approved": False, "approver": "ana"})
        await api.settle()
        job = await api.job(job_id)
        channel = await api.redis.lrange(PUBLISHED_KEY, 0, -1)

    assert job["status"] == "DONE"
    assert job["published"] is False
    assert job["final_answer"] == "El pool de pagos-api se agotó."
    assert channel == []


async def test_an_error_while_running_fails_the_job_with_its_message() -> None:
    async with running_api([RuntimeError("Groq no responde")]) as api:
        job_id = await api.submit()
        await api.settle()
        job = await api.job(job_id)

    assert job["status"] == "FAILED"
    assert "Groq no responde" in job["error"]
    assert job["finished_at"] is not None


async def test_approval_is_refused_for_unknown_jobs_and_jobs_not_waiting_for_one() -> None:
    async with running_api([RuntimeError("caído")]) as api:
        job_id = await api.submit()
        await api.settle()
        approval = {"approved": True, "approver": "ana"}

        unknown = await api.client.post("/tasks/no-existe/approve", json=approval)
        failed = await api.client.post(f"/tasks/{job_id}/approve", json=approval)
        missing = await api.client.get("/tasks/no-existe")

    assert unknown.status_code == 404
    assert missing.status_code == 404
    assert failed.status_code == 409
    assert failed.json()["detail"]["status"] == "FAILED"


async def test_only_the_first_of_two_simultaneous_approvals_counts() -> None:
    async with running_api(finished()) as api:
        job_id = await api.submit()
        await api.settle()

        responses = await asyncio.gather(
            api.client.post(f"/tasks/{job_id}/approve", json={"approved": True, "approver": "ana"}),
            api.client.post(f"/tasks/{job_id}/approve", json={"approved": False, "approver": "beto"}),
        )
        await api.settle()
        job = await api.job(job_id)

    assert sorted(r.status_code for r in responses) == [202, 409]
    winner = next(r for r in responses if r.status_code == 202)
    assert job["approval"]["approver"] == winner.json()["approval"]["approver"]


async def test_jobs_beyond_the_concurrency_limit_wait_in_pending() -> None:
    script: list[AIMessage | Exception | Stall] = [
        Stall(0.5, decide("FINISH")),
        AIMessage(content="Primera."),
        decide("FINISH"),
        AIMessage(content="Segunda."),
    ]
    async with running_api(script, max_concurrent=1) as api:
        first = await api.submit()
        second = await api.submit()
        await asyncio.sleep(0.2)
        while_busy = [(await api.job(first))["status"], (await api.job(second))["status"]]
        await api.settle()
        drafts = [(await api.job(first))["draft"], (await api.job(second))["draft"]]

    assert while_busy == ["RUNNING", "PENDING"]
    assert drafts == ["Primera.", "Segunda."]


async def test_startup_fails_jobs_a_restart_interrupted_but_keeps_paused_ones() -> None:
    redis = FakeRedis(decode_responses=True)
    jobs = JobStore(redis)
    running = await jobs.create(QUESTION)
    await jobs.claim(running.id, "PENDING", "RUNNING")
    waiting = await jobs.create(QUESTION)
    paused = await jobs.create(QUESTION)
    await jobs.claim(paused.id, "PENDING", "PAUSED_FOR_APPROVAL")

    async with running_api([], redis=redis) as api:
        statuses = {job_id: (await api.job(job_id)) for job_id in (running.id, waiting.id, paused.id)}

    assert statuses[running.id]["status"] == "FAILED"
    assert "reinicio" in statuses[running.id]["error"]
    assert statuses[waiting.id]["status"] == "FAILED"
    assert statuses[paused.id]["status"] == "PAUSED_FOR_APPROVAL"


async def test_job_records_its_tokens_and_cost() -> None:
    def billed(message: AIMessage) -> AIMessage:
        message.usage_metadata = {"input_tokens": 250_000, "output_tokens": 50_000, "total_tokens": 300_000}
        message.response_metadata = {"model_name": "openai/gpt-oss-20b"}
        return message

    async with running_api([billed(m) for m in finished() if isinstance(m, AIMessage)]) as api:
        job_id = await api.submit()
        await api.settle()
        job = await api.job(job_id)

    assert job["usage"]["input_tokens"] == 1_000_000
    assert job["usage"]["output_tokens"] == 200_000
    assert job["usage"]["cost_usd"] == pytest.approx(0.135)


async def test_question_is_required() -> None:
    async with running_api([]) as api:
        response = await api.client.post("/tasks", json={"question": ""})

    assert response.status_code == 422
