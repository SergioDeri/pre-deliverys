from typing import Any

import groq
from langgraph.errors import NodeTimeoutError
from langgraph.types import RetryPolicy, TimeoutPolicy

STEP_TIMEOUT = 120.0
RATE_LIMIT_WAIT = 5.0


def stall_guard() -> dict[str, Any]:
    return {
        "timeout": TimeoutPolicy(run_timeout=STEP_TIMEOUT),
        "retry_policy": [
            RetryPolicy(max_attempts=2, retry_on=NodeTimeoutError),
            RetryPolicy(
                max_attempts=6,
                initial_interval=RATE_LIMIT_WAIT,
                max_interval=12 * RATE_LIMIT_WAIT,
                retry_on=groq.RateLimitError,
            ),
        ],
    }
