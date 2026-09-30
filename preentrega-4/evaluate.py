import statistics
from collections.abc import Callable, Iterable, Sequence, Set
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from pydantic import TypeAdapter

from config import DATA_DIR, GOLDEN_DATASET, chunking_settings, connect, hybrid_weights, run
from hybrid_retriever import HybridRetrieval, in_memory_retrieval, pinecone_retrieval
from ingest import load_directory
from schemas import ChunkingSettings, GoldenCase, HybridWeights, RetrievalScores

CUTOFFS = (3, 5)
VECTOR_WEIGHTS = (0.3, 0.5, 0.7)

Retriever = Runnable[str, list[Document]]
Metric = Callable[[Sequence[str], Set[str], int], float]


def retrieved_sources(documents: Iterable[Document]) -> list[str]:
    return list(dict.fromkeys(str(doc.metadata["source"]) for doc in documents))


def precision_at_k(retrieved: Sequence[str], expected: Set[str], k: int) -> float:
    hits = sum(1 for source in retrieved[:k] if source in expected)
    return hits / k


def recall_at_k(retrieved: Sequence[str], expected: Set[str], k: int) -> float:
    found = {source for source in retrieved[:k] if source in expected}
    return len(found) / len(expected)


def load_golden_dataset(path: Path = GOLDEN_DATASET) -> list[GoldenCase]:
    return TypeAdapter(list[GoldenCase]).validate_json(path.read_bytes())


def check_sources(cases: Iterable[GoldenCase], indexed: Set[str]) -> None:
    unknown = sorted({source for case in cases for source in case.expected_sources} - indexed)
    if unknown:
        raise ValueError(f"El Golden Dataset espera Sources que no están indexadas: {', '.join(unknown)}")


def rank(retriever: Retriever, cases: Sequence[GoldenCase]) -> list[list[str]]:
    return [retrieved_sources(retriever.invoke(case.question)) for case in cases]


def score(
    rankings: Sequence[Sequence[str]], cases: Sequence[GoldenCase], cutoffs: Sequence[int] = CUTOFFS
) -> list[RetrievalScores]:
    def mean(metric: Metric, k: int) -> float:
        pairs = zip(rankings, cases, strict=True)
        return sum(metric(ranking, case.expected_sources, k) for ranking, case in pairs) / len(cases)

    return [
        RetrievalScores(k=k, precision=mean(precision_at_k, k), recall=mean(recall_at_k, k))
        for k in cutoffs
    ]


def retrievers_to_compare(retrieval: HybridRetrieval, default: HybridWeights) -> dict[str, Retriever]:
    retrievers: dict[str, Retriever] = {
        "solo vectorial": retrieval.vector(),
        "solo BM25": retrieval.lexical(),
    }
    for vector in sorted({*VECTOR_WEIGHTS, default.vector}):
        weights = HybridWeights(vector=vector)
        retrievers[weights.label] = retrieval.hybrid(weights)
    return retrievers


def print_table(scores_by_config: dict[str, list[RetrievalScores]]) -> None:
    width = max(len("configuración"), *(len(name) for name in scores_by_config))
    header = f"{'configuración':<{width}}" + "".join(f"  {'P@' + str(k):>6}  {'R@' + str(k):>6}" for k in CUTOFFS)
    print(header)
    print("-" * len(header))
    for name, scores in scores_by_config.items():
        print(f"{name:<{width}}" + "".join(f"  {s.precision:>6.3f}  {s.recall:>6.3f}" for s in scores))


def print_misses(rankings: Sequence[Sequence[str]], cases: Sequence[GoldenCase], k: int) -> None:
    for ranking, case in zip(rankings, cases, strict=True):
        top = ranking[:k]
        missing = sorted(case.expected_sources.difference(top))
        if missing:
            print(f"  «{case.question}»")
            print(f"      faltan: {', '.join(missing)}")
            print(f"      top-{k}:  {', '.join(top)}")


def evaluate(retrieval: HybridRetrieval, cases: Sequence[GoldenCase], weights: HybridWeights) -> None:
    check_sources(cases, retrieval.sources())
    print(f"Evaluación sobre {len(cases)} preguntas del Golden Dataset (relevancia por Source)\n")
    rankings = {name: rank(retriever, cases) for name, retriever in retrievers_to_compare(retrieval, weights).items()}
    print_table({name: score(ranked, cases) for name, ranked in rankings.items()})

    print(f"\nPreguntas que el {weights.label} no resuelve en el top-{max(CUTOFFS)}:")
    print_misses(rankings[weights.label], cases, max(CUTOFFS))


def compare_chunking(
    embeddings: Embeddings,
    cases: Sequence[GoldenCase],
    settings: ChunkingSettings,
    weights: HybridWeights,
) -> None:
    encoding = tiktoken.get_encoding("cl100k_base")
    print(f"\nChunking por sección vs. secciones fusionadas hasta {settings.chunk_tokens} tokens")
    print("(en memoria, con los mismos embeddings de Gemini)\n")

    scores_by_config: dict[str, list[RetrievalScores]] = {}
    for variant, merged in (("por sección", False), ("fusionado", True)):
        chunks = load_directory(DATA_DIR, settings, merged=merged)
        tokens = [len(encoding.encode(chunk.text)) for chunk in chunks]
        print(f"  {variant}: {len(chunks)} chunks, mediana {statistics.median(tokens):.0f} tokens, máximo {max(tokens)}")
        retrieval = in_memory_retrieval(chunks, embeddings)
        for name, retriever in (
            ("vectorial", retrieval.vector()),
            ("BM25", retrieval.lexical()),
            (weights.label, retrieval.hybrid(weights)),
        ):
            scores_by_config[f"{variant}: {name}"] = score(rank(retriever, cases), cases)
    print()
    print_table(scores_by_config)


def main() -> None:
    index, embeddings = connect()
    cases = load_golden_dataset()
    weights = hybrid_weights()
    evaluate(pinecone_retrieval(index, embeddings), cases, weights)
    compare_chunking(embeddings, cases, chunking_settings(), weights)


if __name__ == "__main__":
    run(main)
