"""Postgres + pgvector storage: documents, chunks with embeddings and a German full-text index."""

from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from energy_agent.rag.chunking import Chunk
from energy_agent.rag.embeddings import EMBEDDING_DIM
from energy_agent.sources import Source

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    id          text PRIMARY KEY,
    title       text NOT NULL,
    url         text NOT NULL,
    valid_from  date NOT NULL,
    sha256      text NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id           text PRIMARY KEY,
    doc_id       text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index  int  NOT NULL,
    section      text NOT NULL,
    page_start   int  NOT NULL,
    page_end     int  NOT NULL,
    content      text NOT NULL,
    embedding    vector({EMBEDDING_DIM}) NOT NULL,
    -- Section titles weigh more than body text in full-text ranking.
    content_tsv  tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('german', section), 'A') ||
        setweight(to_tsvector('german', content), 'B')
    ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (content_tsv);
"""

_RESULT_COLUMNS = """
    c.id, c.doc_id, d.title, c.section, c.page_start, c.page_end, c.content
"""


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    page_start: int
    page_end: int
    content: str
    score: float


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def replace_document(
    conn: psycopg.Connection,
    source: Source,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    """Store a document and its chunks, replacing any previous version (idempotent)."""
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO documents (id, title, url, valid_from, sha256)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title, url = EXCLUDED.url,
                valid_from = EXCLUDED.valid_from, sha256 = EXCLUDED.sha256
            """,
            (source.id, source.title, source.url, source.valid_from, source.sha256),
        )
        conn.execute("DELETE FROM chunks WHERE doc_id = %s", (source.id,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (id, doc_id, chunk_index, section, page_start, page_end, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        c.id,
                        c.doc_id,
                        c.chunk_index,
                        c.section,
                        c.page_start,
                        c.page_end,
                        c.text,
                        np.asarray(vector, dtype=np.float32),
                    )
                    for c, vector in zip(chunks, embeddings, strict=True)
                ],
            )


def vector_search(
    conn: psycopg.Connection, query_embedding: list[float], k: int
) -> list[SearchResult]:
    rows = conn.execute(
        f"""
        SELECT {_RESULT_COLUMNS}, 1 - (c.embedding <=> %(q)s) AS score
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {"q": np.asarray(query_embedding, dtype=np.float32), "k": k},
    ).fetchall()
    return [SearchResult(*row) for row in rows]


def text_search(conn: psycopg.Connection, query: str, k: int) -> list[SearchResult]:
    # plainto_tsquery ANDs all words, which rarely matches a whole question;
    # OR-ing the stemmed words and ranking by coverage works better for retrieval.
    rows = conn.execute(
        f"""
        WITH q AS (
            SELECT replace(plainto_tsquery('german', %(q)s)::text, '&', '|')::tsquery AS query
        )
        SELECT {_RESULT_COLUMNS}, ts_rank_cd(c.content_tsv, q.query) AS score
        FROM chunks c JOIN documents d ON d.id = c.doc_id, q
        WHERE c.content_tsv @@ q.query
        ORDER BY score DESC
        LIMIT %(k)s
        """,
        {"q": query, "k": k},
    ).fetchall()
    return [SearchResult(*row) for row in rows]


def count_chunks(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
