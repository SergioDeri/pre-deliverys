# Evidence travels as structured state, not as text

The Analyst has to count errors per service, order events in time and measure how long an incident lasted. If all it received were the Researcher's Contribution, those numbers would come from a model reading prose and doing arithmetic on it, which it gets wrong often enough to matter in an incident report. So the Researcher's tools return, next to the text the model reads, the Log Lines and Error Codes they found as Pydantic objects; the graph collects them into an `evidence` field of the shared state, and the Analyst's tools compute over that field (injected with `InjectedState`), never over what a model wrote. The model chooses what to calculate and interprets the result; the code does the calculating.

This also makes "the Analyst works after the Researcher" a real data dependency instead of just a sentence in the Supervisor's prompt.

The model is Groq's `openai/gpt-oss-120b`, for the reasons in [Pre-entrega 5's ADR](../../../preentrega-5/docs/adr/0001-groq-instead-of-openai.md).

## Considered Options

- **Pass only the Contribution text to the Analyst**: simplest, and the usual shape of supervisor examples, but the Analyst's numbers would be unverifiable.
- **Let the Analyst's tools read the logs again from disk**: accurate, but it duplicates the Researcher's job and the two roles stop being distinct.
- **Put the Researcher's ToolMessages in the shared `messages`**: the Analyst could read raw lines, but so would the Supervisor on every turn, which is exactly the context pollution the isolation is meant to avoid.
