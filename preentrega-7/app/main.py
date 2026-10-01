import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, cast

from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.graph import build_graph, create_llm
from app.hitl import RedisIncidentChannel
from app.jobs import DEFAULT_TTL, JobStore
from app.observability import setup_tracing
from app.schemas import Approval, Job
from app.worker import JobRunner


@dataclass
class Services:
    jobs: JobStore
    runner: JobRunner


type ServicesFactory = Callable[[], AbstractAsyncContextManager[Services]]


class NewJob(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def not_found(job_id: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el trabajo {job_id}.")


def create_app(services: ServicesFactory) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with services() as provided:
            await provided.jobs.fail_interrupted()
            app.state.services = provided
            try:
                yield
            finally:
                await provided.runner.close()

    app = FastAPI(title="Orquestador de incidentes de PayFlow", lifespan=lifespan)

    def get_services(request: Request) -> Services:
        return cast(Services, request.app.state.services)

    Deps = Annotated[Services, Depends(get_services)]

    @app.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
    async def submit(body: NewJob, deps: Deps) -> Job:
        job = await deps.jobs.create(body.question)
        deps.runner.start(job.id, body.question)
        return job

    @app.get("/tasks/{job_id}")
    async def read(job_id: str, deps: Deps) -> Job:
        job = await deps.jobs.get(job_id)
        if job is None:
            raise not_found(job_id)
        return job

    @app.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED)
    async def approve(job_id: str, approval: Approval, deps: Deps) -> Job:
        job = await deps.jobs.claim(job_id, "PAUSED_FOR_APPROVAL", "PENDING")
        if job is None:
            current = await deps.jobs.get(job_id)
            if current is None:
                raise not_found(job_id)
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                {"message": "El trabajo no está esperando una aprobación.", "status": current.status},
            )
        job.approval = approval
        job.decided_at = datetime.now(UTC)
        await deps.jobs.save(job)
        deps.runner.resume(job_id, approval)
        return job

    return app


@asynccontextmanager
async def redis_services() -> AsyncIterator[Services]:
    load_dotenv(find_dotenv(usecwd=True))
    tracing = setup_tracing()
    redis_url = os.getenv("REDIS_URL") or "redis://localhost:6379"
    ttl = timedelta(days=float(days)) if (days := os.getenv("JOB_TTL_DAYS")) else DEFAULT_TTL
    redis = Redis.from_url(redis_url, decode_responses=True)
    checkpoint_ttl = {"default_ttl": ttl.total_seconds() / 60, "refresh_on_read": True}
    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=checkpoint_ttl) as checkpoints:
        await checkpoints.asetup()
        jobs = JobStore(redis, ttl)
        graph = build_graph(create_llm(), checkpoints, RedisIncidentChannel(redis))
        yield Services(jobs, JobRunner(graph, jobs, int(os.getenv("MAX_CONCURRENT_JOBS") or 5)))
    await redis.aclose()
    if tracing:
        tracing.shutdown()


app = create_app(redis_services)
