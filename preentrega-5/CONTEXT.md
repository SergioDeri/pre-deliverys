# On-call Agent with Persistent Memory

An agent that helps whoever is on call at PayFlow: it reasons about a question, decides on its own which tools to consult (error catalog, runbooks, logs), and remembers earlier turns of the same conversation across runs.

## Language

### Conversation

**Thread**:
One conversation with the agent, identified by its thread id. Everything said in a Thread is remembered by later Turns of that same Thread, even after the program restarts; nothing crosses from one Thread to another.
_Avoid_: Session, chat, history

**Turn**:
One question from the user and the final answer the agent gives to it, with every Step in between.
_Avoid_: Interaction, request, exchange

**Step**:
One pass through a node of the graph inside a Turn: either the model deciding (answer or Tool Calls) or the tools running.
_Avoid_: Iteration, hop

**Tool Call**:
The model's decision to run one tool with specific arguments. The model makes it; nothing in the code chooses a tool for it.
_Avoid_: Action, function call

**Trace**:
The recorded Steps of every Turn in a run: Tool Calls with their arguments, what each tool returned, and the final answers.
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
