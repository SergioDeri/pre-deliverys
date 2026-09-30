# Validated Extraction Pipeline

Turns unstructured technical text (an architecture description or an error log) into a validated Technical Analysis, or into a controlled failure when the text cannot support one.

## Language

**Source Text**:
The unstructured text the pipeline receives: an architecture description, an error log, or something too vague to be either.
_Avoid_: Prompt, input, document

**Technical Analysis**:
The validated object extracted from a Source Text: its input kind, the technologies it mentions, the affected components, a criticality level and a technical summary.
_Avoid_: Entity, report, extraction result

**Input Kind**:
What the Source Text is: `arquitectura`, `log_error` or `ambiguo`. A `log_error` always names at least one affected component, and an `ambiguo` text can never be of `alta` criticality.
_Avoid_: Type, category

**Technologies**:
The concrete tools, languages, frameworks or services named in the Source Text. A Technical Analysis always has at least one.
_Avoid_: Stack, tools

**Affected Components**:
The parts of the system the Source Text says are involved or failing.
_Avoid_: Services, modules

**Criticality Level**:
How urgent the situation described is: `baja`, `media` or `alta`, and nothing else.
_Avoid_: Severity, priority

**Technical Summary**:
A short descriptive account of what the Source Text says, written for an engineer.
_Avoid_: Description, abstract

**Extraction Outcome**:
What the pipeline always returns: either a Technical Analysis, or an Extraction Failure. It is never an exception.
_Avoid_: Response, result

**Extraction Failure**:
The reason no Technical Analysis could be produced after every attempt was used up: invalid output, truncated output, or a Provider error.
_Avoid_: Error, exception

**Correction**:
A new attempt that tells the model why its previous reply was rejected, so it can fix that reply instead of repeating it. Provider errors are retried without a Correction.
_Avoid_: Retry, feedback loop
