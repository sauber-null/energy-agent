"""Regression gates on the evaluation.

- Retrieval (integration marker, no LLM): hit@5 must not drop below the measured baseline.
- End to end (eval marker, opt-in because it calls a real LLM):
  `uv run pytest -m eval` runs the golden set on the configured provider and fails if the
  pass rate drops below the threshold.
"""

import pytest

from energy_agent.agent import Agent
from energy_agent.config import get_settings
from energy_agent.evaluation.golden import EVAL_TODAY, load_golden
from energy_agent.evaluation.retrieval import retrieval_metrics
from energy_agent.evaluation.runner import run_golden, summarize
from energy_agent.llm import create_chat_model
from energy_agent.rag.embeddings import OllamaEmbedder
from energy_agent.rag.search import Retriever
from energy_agent.rag.store import count_chunks
from energy_agent.tools.registry import build_tools
from energy_agent.tools.switch_requests import seed

# Baselines measured on 2026-10-27 (see README, Evaluation), with a small margin.
MIN_HIT_AT_5 = {"vector": 0.7, "hybrid": 0.6}
MIN_PASS_RATE = 0.8  # Mistral ministral-8b-latest: 0.86-0.90


@pytest.fixture(scope="module")
def retriever(db_conn):
    try:
        if count_chunks(db_conn) == 0:
            pytest.skip("no chunks ingested")
    except Exception as exc:
        db_conn.rollback()
        pytest.skip(f"chunks table not available: {exc}")
    settings = get_settings()
    return Retriever(db_conn, OllamaEmbedder(settings.ollama_host, settings.embedding_model))


@pytest.mark.integration
def test_retrieval_hit_rate(retriever):
    metrics = retrieval_metrics(retriever, load_golden(), modes=tuple(MIN_HIT_AT_5))
    for mode, minimum in MIN_HIT_AT_5.items():
        assert metrics[mode]["hit@5"] >= minimum, metrics


@pytest.mark.eval
def test_golden_set_pass_rate(db_conn, retriever):
    settings = get_settings()
    seed(db_conn)
    agent = Agent(
        create_chat_model(settings),
        build_tools(db_conn, retriever),
        today=EVAL_TODAY,
    )
    results = run_golden(agent, load_golden())
    summary = summarize(results)
    failed = {r.id: r.failures for r in results if not r.passed}
    assert summary["pass_rate"] >= MIN_PASS_RATE, failed
