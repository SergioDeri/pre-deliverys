# Gemini embeddings computed outside Chroma

The collection has no embedding function: we call `gemini-embedding-001` through the `google-genai` SDK ourselves and hand Chroma the vectors (`embeddings=` on upsert, `query_embeddings=` on query). Gemini embeds chunks with `task_type=RETRIEVAL_DOCUMENT` and queries with `RETRIEVAL_QUERY`, which improves retrieval, and a Chroma embedding function can only apply one task type to both. The Embedding Space (model and dimension, 768) is stored in the collection metadata, and opening the collection under a different one fails instead of mixing vectors.

## Considered Options

- **Chroma's `GoogleGenerativeAiEmbeddingFunction`**: rejected because it uses the same task type for chunks and queries.
- **`gemini-embedding-2`**: rejected because it ignores `task_type` (it returned the same vector for both in our test), so the asymmetry would be lost.
- **3072 dimensions (the default)**: rejected as unnecessary for a small corpus. Below 3072 Gemini does not normalize its vectors, so we normalize them before storing them in a cosine-space collection.
