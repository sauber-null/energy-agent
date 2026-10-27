"""Command line entry point: `energy-agent <command>`."""

import argparse
import json
from dataclasses import asdict

from energy_agent.config import get_settings


def cmd_info(args: argparse.Namespace) -> None:
    settings = get_settings()
    print(f"provider={settings.llm_provider}, model={settings.active_model}")


def cmd_chunk(args: argparse.Namespace) -> None:
    from energy_agent.rag.chunking import chunk_document
    from energy_agent.rag.pdf import pdf_to_markdown_pages
    from energy_agent.sources import PROCESSED_DIR, load_sources

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "chunks.jsonl"
    with out.open("w") as f:
        for source in load_sources():
            chunks = chunk_document(source.id, pdf_to_markdown_pages(source.path), args.max_chars)
            for chunk in chunks:
                f.write(json.dumps({"id": chunk.id, **asdict(chunk)}, ensure_ascii=False) + "\n")
            sizes = [len(c.text) for c in chunks]
            print(
                f"{source.id}: {len(chunks)} chunks, "
                f"avg {sum(sizes) // len(sizes)} chars, max {max(sizes)}"
            )
    print(f"written to {out}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from energy_agent.rag.embeddings import OllamaEmbedder
    from energy_agent.rag.ingest import ingest
    from energy_agent.rag.store import connect, count_chunks
    from energy_agent.sources import load_sources

    settings = get_settings()
    embedder = OllamaEmbedder(settings.ollama_host, settings.embedding_model)
    with connect(settings.database_url) as conn:
        ingest(conn, embedder, load_sources(), args.max_chars)
        print(f"{count_chunks(conn)} chunks in the database")


def cmd_search(args: argparse.Namespace) -> None:
    from energy_agent.rag.embeddings import OllamaEmbedder
    from energy_agent.rag.search import Retriever
    from energy_agent.rag.store import connect

    settings = get_settings()
    embedder = OllamaEmbedder(settings.ollama_host, settings.embedding_model)
    with connect(settings.database_url) as conn:
        results = Retriever(conn, embedder).search(args.query, args.k, args.mode)
    for i, r in enumerate(results, start=1):
        pages = f"S. {r.page_start}" + (f"-{r.page_end}" if r.page_end != r.page_start else "")
        print(f"\n[{i}] {r.score:.4f}  {r.doc_id}, {pages}\n    {r.section}")
        print("    " + r.content[:300].replace("\n", "\n    "))


def main() -> None:
    parser = argparse.ArgumentParser(prog="energy-agent")
    commands = parser.add_subparsers(required=True)

    commands.add_parser("info", help="show the active LLM provider and model").set_defaults(
        func=cmd_info
    )

    chunk = commands.add_parser("chunk", help="parse the PDFs into data/processed/chunks.jsonl")
    chunk.add_argument("--max-chars", type=int, default=1500)
    chunk.set_defaults(func=cmd_chunk)

    ingest = commands.add_parser("ingest", help="parse, embed and store all source documents")
    ingest.add_argument("--max-chars", type=int, default=1500)
    ingest.set_defaults(func=cmd_ingest)

    search = commands.add_parser("search", help="search the document chunks")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--mode", choices=["hybrid", "vector", "text"], default="hybrid")
    search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)
