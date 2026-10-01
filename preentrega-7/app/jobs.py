import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.schemas import Job, JobStatus

DEFAULT_TTL = timedelta(days=7)
INTERRUPTED = "Se cortó por un reinicio de la API mientras el trabajo estaba en curso."


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


class JobStore:
    def __init__(self, redis: Redis, ttl: timedelta = DEFAULT_TTL) -> None:
        self.redis = redis
        self.ttl = ttl

    async def create(self, question: str) -> Job:
        job = Job(id=uuid4().hex, status="PENDING", question=question, created_at=datetime.now(UTC))
        await self.save(job)
        return job

    async def get(self, job_id: str) -> Job | None:
        fields = await self.redis.hgetall(job_key(job_id))
        if not fields:
            return None
        return Job.model_validate({**json.loads(fields["data"]), "status": fields["status"]})

    async def save(self, job: Job) -> None:
        key = job_key(job.id)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping={"status": job.status, "data": job.model_dump_json()})
            pipe.expire(key, self.ttl)
            await pipe.execute()

    async def claim(self, job_id: str, expected: JobStatus, new: JobStatus) -> Job | None:
        key = job_key(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                if await pipe.hget(key, "status") != expected:
                    return None
                pipe.multi()  # type: ignore[no-untyped-call]
                pipe.hset(key, "status", new)
                await pipe.execute()
            except WatchError:
                return None
        return await self.get(job_id)

    async def fail_interrupted(self) -> list[str]:
        failed = []
        async for key in self.redis.scan_iter(match=job_key("*")):
            job_id = key.removeprefix(job_key(""))
            for status in ("PENDING", "RUNNING"):
                job = await self.claim(job_id, status, "FAILED")
                if job is not None:
                    job.fail(INTERRUPTED)
                    await self.save(job)
                    failed.append(job_id)
        return failed
