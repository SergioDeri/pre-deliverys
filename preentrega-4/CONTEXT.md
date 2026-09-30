# Hybrid Retrieval in the Cloud

Ingests local documents as embedded chunks into a cloud vector index, answers queries by combining semantic and lexical retrieval, and measures how well retrieval finds the expected sources.

## Language

### Ingestion

**Source**:
One local file under `data/` (markdown, text, log or JSON), identified by its path relative to `data/`.
_Avoid_: Document, file, input

**Category**:
What kind of knowledge a Source holds: `arquitectura`, `runbook`, `log` or `errores`. It comes from the folder the Source lives in, never from its extension.
_Avoid_: Type, kind, tag

**Error Code**:
One entry of the error catalog (`PF-4012`): its meaning, cause and the runbook to follow. Each Error Code is exactly one Chunk.
_Avoid_: Error, record, entry

**Section**:
Where inside its Source a Chunk sits: the markdown heading it falls under, or its Error Code. Other Sources have no Section.
_Avoid_: Header, chapter

**Namespace**:
The partition of the index that holds every Chunk of one Category. A search either stays in one Namespace or spans all of them.
_Avoid_: Collection, partition, tenant

**Chunk**:
A slice of a Source, and the unit that is embedded, stored and retrieved. It carries its own text, Source, Category and Section, so nothing else has to be looked up to show it.
_Avoid_: Fragment, passage, document

**Embedding Space**:
The pair of embedding model and vector dimension the index was built with. An index refuses to be used under a different one.
_Avoid_: Model config, vector config

### Retrieval

**Vector Retrieval**:
Ranking Chunks by closeness in meaning to the query.
_Avoid_: Dense search, semantic search

**Lexical Retrieval**:
Ranking Chunks by exact term overlap with the query (BM25), so codes, service names and proper nouns match literally.
_Avoid_: Keyword search, sparse search, full-text

**Hybrid Retrieval**:
One ranking built by fusing Vector Retrieval and Lexical Retrieval, each with a weight.
_Avoid_: Ensemble search, combined search

### Evaluation

**Golden Dataset**:
The fixed set of test questions, each paired with the Expected Sources that a good retrieval must find.
_Avoid_: Test set, ground truth, benchmark

**Expected Sources**:
The Sources that answer a Golden Dataset question. Relevance is judged per Source, not per Chunk.
_Avoid_: Expected documents, relevant chunks, labels

**Retrieved Sources**:
The ranked Chunks of a retrieval collapsed to distinct Sources, keeping the order in which each Source first appears.
_Avoid_: Results, hits

**Precision@k**:
Of the first k Retrieved Sources, the share that are Expected Sources.
_Avoid_: Accuracy

**Recall@k**:
Of the Expected Sources, the share that appear among the first k Retrieved Sources.
_Avoid_: Coverage, hit rate
