"""Ingestion pipeline: PDF -> chunks -> embeddings -> Postgres."""

import time
from collections.abc import Callable

import psycopg

from energy_agent.rag import store
from energy_agent.rag.chunking import Chunk, chunk_document
from energy_agent.rag.embeddings import OllamaEmbedder
from energy_agent.rag.pdf import pdf_to_markdown_pages
from energy_agent.sources import Source


def embedding_text(doc_title: str, chunk: Chunk) -> str:
    # A chunk on its own often lacks the process it belongs to ("Frist: unverzüglich ..."),
    # so the document title and section path are embedded together with the text.
    return f"{doc_title}\n{chunk.section}\n\n{chunk.text}"


def ingest(
    conn: psycopg.Connection,
    embedder: OllamaEmbedder,
    sources: list[Source],
    max_chars: int = 1500,
    log: Callable[[str], None] = print,
) -> None:
    store.init_schema(conn)
    for source in sources:
        started = time.perf_counter()
        chunks = chunk_document(source.id, pdf_to_markdown_pages(source.path), max_chars)
        embeddings = embedder.embed([embedding_text(source.title, c) for c in chunks])
        store.replace_document(conn, source, chunks, embeddings)
        log(f"{source.id}: {len(chunks)} chunks in {time.perf_counter() - started:.1f}s")
