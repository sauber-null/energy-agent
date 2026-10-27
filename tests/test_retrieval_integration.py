"""Retrieval smoke tests on the real corpus.

Needs `docker compose up -d` and `energy-agent ingest`; skipped otherwise.
"""

import pytest

from energy_agent.config import get_settings
from energy_agent.rag.embeddings import OllamaEmbedder
from energy_agent.rag.search import Retriever
from energy_agent.rag.store import count_chunks

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def retriever(db_conn):
    try:
        if count_chunks(db_conn) == 0:
            pytest.skip("no chunks ingested")
    except Exception as exc:  # schema missing
        db_conn.rollback()
        pytest.skip(f"chunks table not available: {exc}")
    settings = get_settings()
    return Retriever(db_conn, OllamaEmbedder(settings.ollama_host, settings.embedding_model))


@pytest.mark.parametrize(
    ("query", "expected_section"),
    [
        ("Wer beliefert einen Kunden in der Ersatzversorgung?", "Ersatz-/Grundversorgung"),
        ("Wie schnell muss der Netzbetreiber auf eine MaLo-ID-Anfrage antworten?", "MaLo-ID"),
        ("Kündigung des bisherigen Liefervertrags durch den neuen Lieferanten", "Kündigung"),
    ],
)
@pytest.mark.parametrize("mode", ["hybrid", "vector"])
def test_expected_section_is_in_top_5(retriever, query, expected_section, mode):
    results = retriever.search(query, k=5, mode=mode)
    assert any(expected_section in r.section for r in results), [r.section for r in results]
