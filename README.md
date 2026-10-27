# energy-agent

A tool-calling AI agent for German electricity market processes (GPKE: supplier switch,
start/end of supply, default supply). It answers questions with retrieval over public
BNetzA documents, calculates process deadlines and looks up switching requests in a test
database. The same agent runs against a hosted model (Mistral) and a local model (Ollama).

> Work in progress. Architecture, setup and evaluation results follow as the project grows.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (installs Python 3.12 automatically), Docker
(e.g. OrbStack) and [Ollama](https://ollama.com).

```bash
uv sync
cp .env.example .env                        # then add your Mistral API key
ollama pull bge-m3                          # local embedding model
docker compose up -d                        # Postgres 17 + pgvector
uv run python scripts/download_sources.py   # BNetzA PDFs -> data/raw/
uv run energy-agent ingest                  # parse, chunk, embed, store (~1 min)
uv run energy-agent search "Wer beliefert einen Kunden in der Ersatzversorgung?"
uv run pytest                               # unit + integration tests
```

## Retrieval (RAG)

```
PDF ──PyMuPDF──▶ Markdown pages ──▶ section-aware chunks ──bge-m3 (Ollama)──▶ Postgres
                                                                   ├─ vector(1024), HNSW index
                                                                   └─ tsvector('german'), GIN index
query ──▶ vector top-20 ─┐
      └─▶ full-text top-20 ─┴─▶ Reciprocal Rank Fusion ──▶ top-k chunks
```

- **Parsing:** PyMuPDF renders each page as Markdown, so the GPKE process tables
  (`Nr. | Aktion | Frist | Hinweis`) keep their columns. Page footers are removed.
- **Chunking** ([`chunking.py`](src/energy_agent/rag/chunking.py)): split at headings, then
  packed to ~1,500 characters. Each chunk stores its section path
  (`2 Zuordnungsprozesse > 2.1 Use-Case: Lieferbeginn > 2.1.2 SD: Lieferbeginn`) and page range;
  split tables repeat their header row. 4 documents, 231 pages → 464 chunks.
- **Embeddings:** `bge-m3` (multilingual, German-capable) runs locally via Ollama. The document
  title and section path are embedded together with the chunk text.
- **Hybrid search** ([`search.py`](src/energy_agent/rag/search.py)): vector similarity finds
  paraphrases, German full-text search finds exact terms such as `MaLo-ID`; both rankings are
  merged with Reciprocal Rank Fusion, which needs ranks only, not comparable scores.

Note: PyMuPDF is AGPL-licensed, which is fine for this open-source project; a closed-source
product would need its commercial licence or another PDF parser.

## Data

The corpus is the current GPKE as published by the Bundesnetzagentur with decision
[BK6-22-024](https://www.bundesnetzagentur.de/DE/Beschlusskammern/1_GZ/BK6-GZ/2022/BK6-22-024/BK6-22-024_Verfahren.html)
(supplier switch within 24 hours, in force since 2025-06-06): the decision itself and
GPKE parts 1, 2 and 4. The PDFs are not stored in the repo; URLs and SHA-256 checksums are
pinned in [`data/sources.toml`](data/sources.toml).

```bash
uv run python scripts/download_sources.py   # downloads to data/raw/ and verifies checksums
```
