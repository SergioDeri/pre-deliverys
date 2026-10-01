# Jobs run inside the API process, with their record in Redis

A Job runs as an asyncio task that the API process creates and owns (`JobRunner`), not in a separate worker reading a queue. Starlette's `BackgroundTasks` would also run in-process, but it ties the work to the request that started it, so the runner keeps its own tasks to cancel them on shutdown and to wait for them in tests. The graph is fully async, so a running investigation only holds the event loop while it is between awaits, and `POST /tasks` keeps answering in milliseconds with several Jobs in flight. A semaphore (`MAX_CONCURRENT_JOBS`) caps how many run at once; the rest wait in `PENDING`.

What the API reports about a Job lives in its own Redis hash (`job:{id}`), written by the code that runs the graph: status, Delegation Trace so far, draft, result, Approval, error, tokens and cost. The LangGraph checkpoints, also in Redis through `RedisSaver`, hold the graph's state and nothing else; the API never reads them to answer `GET /tasks/{id}`.

The price of running in-process is that a Job dies with the process. On startup the API marks every Job still `PENDING` or `RUNNING` as `FAILED` ("interrupted by a restart") instead of leaving it stuck: both were waiting or working inside the process that died. A Job that was `PAUSED_FOR_APPROVAL` is not affected: its pause is in the checkpoint, and an Approval after the restart resumes it where it stopped.

## Considered Options

- **A worker process with a queue (arq, RQ, Celery)**: survives API restarts and scales separately, but adds a second process to run and deploy for a workload of a handful of Jobs, and the assignment asks for background tasks.
- **Read the Job's status from the checkpoints**: one less thing to store, but checkpoints are LangGraph's internal format, change shape across versions, and do not know about `PENDING`, `FAILED` or cost.
