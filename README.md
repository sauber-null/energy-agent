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
