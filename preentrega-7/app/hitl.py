import json
from datetime import UTC, datetime
from typing import Any, Protocol

from langgraph.types import interrupt
from redis.asyncio import Redis

from app.schemas import Approval
from app.state import Node, OrchestratorState, final_answer

PUBLISHED_KEY = "incidents:published"


class IncidentChannel(Protocol):
    async def publish(self, report: str, approval: Approval) -> None: ...


class RedisIncidentChannel:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def publish(self, report: str, approval: Approval) -> None:
        entry = {
            "report": report,
            "approver": approval.approver,
            "comment": approval.comment,
            "published_at": datetime.now(UTC).isoformat(),
        }
        await self.redis.rpush(PUBLISHED_KEY, json.dumps(entry, ensure_ascii=False))


def publish_node(channel: IncidentChannel) -> Node:
    async def publish(state: OrchestratorState) -> dict[str, Any]:
        draft = final_answer(state)
        approval = Approval.model_validate(interrupt({"draft": draft}))
        if approval.approved:
            await channel.publish(draft, approval)
        return {"approval": approval, "published": approval.approved}

    return publish
