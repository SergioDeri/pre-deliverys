# Local Semantic Retrieval

Indexes local documents as embedded chunks in a persistent vector store, and answers semantic queries over them, optionally narrowed by metadata.

## Language

### Indexing

**Source**:
One local file under `data/` (a markdown document, a text file or a log), identified by its path relative to `data/`.
_Avoid_: Document, file, input

**Category**:
What kind of knowledge a Source holds: `arquitectura`, `runbook` or `log`. It comes from the folder the Source lives in, never from its extension.
_Avoid_: Type, kind, tag

**Section**:
The markdown heading a Chunk falls under. Sources that are not markdown have no Section.
_Avoid_: Header, chapter

**Chunk**:
A contiguous, overlapping slice of a Source, and the unit that is embedded, stored and retrieved. It always carries its Source, Category, Section, position within the Source and ingestion time.
_Avoid_: Fragment, passage, document

**Reindexing**:
Replacing every Chunk of a Source with freshly cut ones, so that none of its old Chunks survive.
_Avoid_: Update, refresh

### Embeddings

**Embedding Space**:
The pair of embedding model and vector dimension a collection was built with. Vectors from different Embedding Spaces are never compared, and a collection refuses to be opened under a different one.
_Avoid_: Model config, vector config

**Document Embedding**:
The vector of a chunk, computed as something to be found.
_Avoid_: Index vector

**Query Embedding**:
The vector of a search query, computed as something that looks for Document Embeddings. It is never stored.
_Avoid_: Search vector

### Retrieval

**Search Query**:
A question in natural language, how many Chunks to return, and optionally a Metadata Filter and a minimum Score.
_Avoid_: Prompt, request

**Metadata Filter**:
A restriction on the Category and/or Source a retrieved Chunk may come from. When both are given, a Chunk must match both.
_Avoid_: Where clause, facet

**Score**:
How close a Chunk is in meaning to a Search Query, where 1 means identical in meaning. It is derived from the cosine distance.
_Avoid_: Relevance, similarity, confidence

**Search Result**:
One retrieved Chunk: its text, its Score, its raw distance and its metadata.
_Avoid_: Match, hit
