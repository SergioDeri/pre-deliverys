# Unified Async LLM Client

A provider-agnostic, non-blocking layer for talking to LLMs: callers send a conversation and get back either a complete response or a stream of text, without knowing which provider is behind it.

## Language

**Provider**:
An external LLM vendor reachable through its own official SDK (currently only Groq).
_Avoid_: Backend, vendor, API

**LLM Client**:
The object that talks to exactly one Provider while exposing the common interface every Provider must honour.
_Avoid_: Manager, wrapper, adapter

**Chat Message**:
One turn of a conversation, made of a role (`system`, `user` or `assistant`) and its text content.
_Avoid_: Prompt, message dict

**Generation Config**:
The sampling parameters and model used for a generation. An LLM Client holds a default one, and a single call can override it.
_Avoid_: Settings, options, params

**Retry Policy**:
The rules that decide how many times, and how far apart, an LLM Client retries a Retryable error.
_Avoid_: Backoff config

**Model Response**:
The result of a non-streamed generation. It is always returned, even on failure, in which case it carries an LLM Error instead of content.
_Avoid_: Completion, result, reply

**LLM Error**:
A structured description of why a generation failed: its kind (auth, rate limit, timeout, network, server, invalid request, unknown), the Provider it came from, and whether it is retryable.
_Avoid_: Exception, failure

**Retryable error**:
An LLM Error whose cause is expected to go away by itself (rate limit, timeout, network, server), so the LLM Client tries again with backoff. Every other error is returned right away.
_Avoid_: Transient error, recoverable error

**Stream**:
A generation delivered as successive text fragments. It is retried only if it fails before the first fragment. A failure after that ends the Stream with an LLM Error, because retrying would duplicate text.
_Avoid_: Streaming response, token feed
