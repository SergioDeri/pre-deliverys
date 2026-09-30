# Gemini embeddings at 1536 dimensions instead of OpenAI

The assignment suggests OpenAI `text-embedding-3-small` (1536 dimensions). We use `gemini-embedding-001` with `output_dimensionality=1536` instead, wrapped in our own LangChain `Embeddings` class. It keeps the free Gemini key the other pre-deliveries already use, keeps the 1536 dimensions the assignment expects for the index, and keeps the asymmetric `task_type` (`RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for questions) from Pre-entrega 3. The model name is stored as a tag on the Pinecone index and the dimension is the index's own, so opening the index under a different Embedding Space fails instead of mixing vectors.

## Considered Options

- **OpenAI `text-embedding-3-small`**: rejected because it needs a paid key that none of the other pre-deliveries use.
- **`GoogleGenerativeAIEmbeddings` from `langchain-google-genai`**: rejected because below 3072 dimensions Gemini does not normalize its vectors, and the index uses cosine; our class normalizes them and caches query vectors, since the vector retriever asks every namespace with the same question.
