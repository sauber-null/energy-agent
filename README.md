# energy-agent

A tool-calling AI agent for German electricity market processes (GPKE: supplier switch,
start/end of supply, default supply). It answers questions with retrieval over public
BNetzA documents, calculates process deadlines and looks up switching requests in a test
database. The same agent runs against a hosted model (Mistral) and a local model (Ollama).

> Work in progress. Architecture, setup and evaluation results follow as the project grows.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed by uv automatically).

```bash
uv sync
cp .env.example .env   # then add your Mistral API key
uv run energy-agent
uv run pytest
```

## Data

The corpus is the current GPKE as published by the Bundesnetzagentur with decision
[BK6-22-024](https://www.bundesnetzagentur.de/DE/Beschlusskammern/1_GZ/BK6-GZ/2022/BK6-22-024/BK6-22-024_Verfahren.html)
(supplier switch within 24 hours, in force since 2025-06-06): the decision itself and
GPKE parts 1, 2 and 4. The PDFs are not stored in the repo; URLs and SHA-256 checksums are
pinned in [`data/sources.toml`](data/sources.toml).

```bash
uv run python scripts/download_sources.py   # downloads to data/raw/ and verifies checksums
```
