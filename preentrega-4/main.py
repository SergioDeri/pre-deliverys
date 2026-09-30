import textwrap

from langchain_core.documents import Document

from config import DATA_DIR, chunking_settings, connect, hybrid_weights, run
from evaluate import compare_chunking, evaluate, load_golden_dataset
from hybrid_retriever import pinecone_retrieval
from ingest import Ingestor, summary
from schemas import Category

ERROR_CODE = "PF-5021"
PARAPHRASE = "¿Qué hacemos si el procesador de tarjetas está lento y se acumulan pagos sin autorizar?"


def show(title: str, docs: list[Document], limit: int = 3) -> None:
    print(f"\n== {title} ==")
    if not docs:
        print("  (sin resultados)")
    for rank, doc in enumerate(docs[:limit], start=1):
        meta = doc.metadata
        where = meta["source"] + (f" › {meta['section']}" if meta.get("section") else "")
        excerpt = textwrap.shorten(" ".join(doc.page_content.split()), width=110)
        print(f"  {rank}. [{meta['category']}] {where}")
        print(f"     {excerpt}")


def main() -> None:
    index, embeddings = connect()
    settings = chunking_settings()
    chunks = Ingestor(index, embeddings, settings, DATA_DIR).ingest_directory()
    print(f"Ingestados {summary(chunks)} en Pinecone")

    retrieval = pinecone_retrieval(index, embeddings)
    weights = hybrid_weights()
    hybrid = weights.label.capitalize()

    show(f"Solo vectorial: {ERROR_CODE}", retrieval.vector().invoke(ERROR_CODE))
    show(f"Solo BM25: {ERROR_CODE}", retrieval.lexical().invoke(ERROR_CODE))
    show(f"{hybrid}: {ERROR_CODE}", retrieval.hybrid(weights).invoke(ERROR_CODE))
    show(f"{hybrid}: {PARAPHRASE}", retrieval.hybrid(weights).invoke(PARAPHRASE))
    show(
        f"{hybrid} solo en el namespace runbook: {PARAPHRASE}",
        retrieval.hybrid(weights, Category.RUNBOOK).invoke(PARAPHRASE),
    )

    print()
    cases = load_golden_dataset()
    evaluate(retrieval, cases, weights)
    compare_chunking(embeddings, cases, settings, weights)


if __name__ == "__main__":
    run(main)
