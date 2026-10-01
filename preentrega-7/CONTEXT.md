# Incident Orchestrator API

The Pre-entrega 6 team of agents, served as an API: whoever is on call submits a question, the team investigates it in the background, and nothing leaves the team until a person approves it.

## Language

### Jobs

**Job**:
One investigation submitted through the API, from the question to its outcome. It has a single id, which is also the id of its conversation in the checkpoints.
_Avoid_: Task, request, run

**Job Status**:
Where a Job is in its life: `PENDING` (accepted, waiting for a free slot), `RUNNING`, `PAUSED_FOR_APPROVAL`, `DONE` or `FAILED`. A rejected Publication still ends in `DONE`.
_Avoid_: State, phase

**Publication**:
Sending the Final Answer to the incidents channel, where other people read it. It is the only thing the team does that leaves the system, so it never happens without an Approval.
_Avoid_: Post, notification, send

**Approval**:
A named person's decision on a pending Publication: approve or reject, optionally with a comment. It is what resumes a Job paused for approval, and only the first Approval of a pause counts.
_Avoid_: Confirmation, review, sign-off

### Orchestration

**Supervisor**:
The agent that reads the user's question and the Contributions so far, and decides who works next or that the work is done. It never consults data itself.
_Avoid_: Router, orchestrator, manager

**Specialist**:
An agent with one job and its own tools, which only acts when the Supervisor delegates to it. There are two: the Researcher and the Analyst.
_Avoid_: Worker, sub-agent, expert

**Researcher**:
The Specialist that gathers Evidence about the incident from PayFlow's sources (logs, error catalog, runbooks, architecture). It reports what it found; it does not draw conclusions from it.
_Avoid_: Searcher, retriever

**Analyst**:
The Specialist that works on the Evidence the Researcher already gathered: counts, timelines, root cause, impact. It never goes back to the sources itself, so it cannot act before the Researcher has.
_Avoid_: Calculator, processor

**Evidence**:
What the Researcher collected, exactly as the source returned it: Log Lines and Error Codes. Its Contribution is what it says about the Evidence; the Evidence itself is never rewritten by a model. Evidence from several Delegations adds up, and the same Log Line or Error Code is never counted twice.
_Avoid_: Data, findings, context

**Delegation**:
One decision of the Supervisor to send a Specialist to work, with the Instruction it must follow and the reason for choosing it.
_Avoid_: Handoff, routing, call

**Instruction**:
The concrete task the Supervisor writes for a Specialist in a Delegation. It is all the Specialist is told, besides earlier Contributions.
_Avoid_: Prompt, query, subtask

**Contribution**:
What a Specialist hands back to the Supervisor when it finishes a Delegation. Only the Contribution leaves the Specialist; how it got there stays inside.
_Avoid_: Result, output, response

**Final Answer**:
The answer the Supervisor writes for the user once it decides the work is done, consolidating every Contribution.
_Avoid_: Synthesis, summary, response

**Delegation Limit**:
The maximum number of Delegations in one run. When it is reached the Supervisor must write the Final Answer with what it has, whatever it would have chosen.
_Avoid_: Max steps, budget

**Override**:
A Delegation to the Analyst made before the Researcher has contributed, which is sent to the Researcher instead with the same Instruction. It is recorded as such in the Delegation Trace.
_Avoid_: Redirect, correction, fallback

**Delegation Trace**:
The ordered record of a run: each Delegation with its Instruction and reason, each Override, each Contribution, and the Final Answer.
_Avoid_: Log, history, transcript

### PayFlow knowledge

**Error Code**:
One entry of the error catalog (`PF-4012`): its title, HTTP status, cause and, if it has one, the Runbook to follow.
_Avoid_: Error, record

**Runbook**:
The written procedure to diagnose and mitigate one kind of incident.
_Avoid_: Guide, playbook, doc

**Log Line**:
One line of a service log: timestamp, level, service and message.
_Avoid_: Event, entry, record
