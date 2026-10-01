import asyncio
import time
from collections.abc import Awaitable, Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler

from app.graph import Orchestrator, StepListener, resume_job, start_job
from app.jobs import JobStore
from app.schemas import Approval, Job, Segment, TraceStep, Usage

PRICES_PER_MILLION = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}

type Advance = Callable[[StepListener, list[BaseCallbackHandler]], Awaitable[Segment]]


class JobRunner:
    def __init__(self, graph: Orchestrator, jobs: JobStore, max_concurrent: int) -> None:
        self.graph = graph
        self.jobs = jobs
        self._slots = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task[None]] = set()

    def start(self, job_id: str, question: str) -> None:
        self._spawn(
            self._run(job_id, lambda listen, callbacks: start_job(self.graph, job_id, question, listen, callbacks))
        )

    def resume(self, job_id: str, approval: Approval) -> None:
        self._spawn(
            self._run(job_id, lambda listen, callbacks: resume_job(self.graph, job_id, approval, listen, callbacks))
        )

    async def join(self) -> None:
        while self._tasks:
            await asyncio.gather(*self._tasks)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def _spawn(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, job_id: str, advance: Advance) -> None:
        async with self._slots:
            job = await self.jobs.get(job_id)
            if job is None:
                return

            async def listen(step: TraceStep) -> None:
                job.steps.append(step)
                await self.jobs.save(job)

            usage = UsageMetadataCallbackHandler()
            clock = time.monotonic()
            try:
                job.status = "RUNNING"
                job.started_at = job.started_at or datetime.now(UTC)
                await self.jobs.save(job)
                segment = await advance(listen, [usage])
            except Exception as error:
                job.fail(f"{type(error).__name__}: {error}")
            else:
                settle(job, segment)
            job.run_seconds += time.monotonic() - clock
            job.usage = add_usage(job.usage, usage)
            await self.jobs.save(job)


def settle(job: Job, segment: Segment) -> None:
    if segment.draft is not None:
        job.status = "PAUSED_FOR_APPROVAL"
        job.draft = segment.draft
        job.paused_at = datetime.now(UTC)
        return
    job.status = "DONE"
    job.final_answer = job.draft or segment.final_answer
    job.published = segment.published
    job.draft = None
    job.finished_at = datetime.now(UTC)


def add_usage(total: Usage, handler: UsageMetadataCallbackHandler) -> Usage:
    result = total.model_copy()
    for model, usage in handler.usage_metadata.items():
        input_price, output_price = PRICES_PER_MILLION.get(model, (0.0, 0.0))
        result.input_tokens += usage["input_tokens"]
        result.output_tokens += usage["output_tokens"]
        result.cost_usd += (usage["input_tokens"] * input_price + usage["output_tokens"] * output_price) / 1_000_000
    return result
