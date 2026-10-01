# Groq instead of OpenAI for the agent

The assignment's `.env.example` asks for `OPENAI_API_KEY`. The agent runs on Groq through `ChatGroq` instead, with `openai/gpt-oss-120b` as the default model, for the same reason as [Pre-entrega 1](../../../preentrega-1/docs/adr/0001-groq-only-provider.md): Groq is the only provider we have a working key for, and an agent that has never run against a real model is worse than one that is absent. The graph only depends on `BaseChatModel.bind_tools`, so switching to `ChatOpenAI` means changing `create_llm` and nothing else.

## Considered Options

- **`ChatOpenAI` as the assignment suggests**: rejected because it needs a paid key that none of the other pre-deliveries use.
- **A smaller Groq model such as `llama-3.1-8b-instant`**: rejected because `gpt-oss-120b` is the model Pre-entrega 2 already uses, supports tool calling, and returns its reasoning in a separate field that the Trace records. `GROQ_MODEL` still overrides it.
