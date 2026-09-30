# Groq as the only Provider

The assignment asks for OpenAI and Anthropic clients built on `AsyncOpenAI` and `AsyncAnthropic`. We ship Groq alone, through its official `AsyncGroq` SDK, because it is the only Provider we have a working API key for (its free tier), and code that has never run against a real API is worse than code that is absent. Interchangeability is still shown by `BaseLLMClient`, the `Provider` enum and the client factory: adding OpenAI or Anthropic later means writing one subclass and registering it, with no change for callers.

## Considered Options

- **Groq through `AsyncOpenAI` with Groq's `base_url`**: rejected because it would make Groq look like an alias of OpenAI rather than a Provider of its own.
- **Writing OpenAI and Anthropic clients without keys to test them**: rejected because they would be untested code.
