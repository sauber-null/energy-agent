"""Retrieval smoke tests on the real corpus.

Needs `docker compose up -d` and `energy-agent ingest`; skipped otherwise.
"""

import pytest

from energy_agent.config import get_settings
from energy_agent.rag.embeddings import OllamaEmbedder
from energy_agent.rag.search import Retriever
from energy_agent.rag.store import connect, count_chunks

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def retriever():
    settings = get_settings()
    try:
        conn = connect(settings.database_url + "?connect_timeout=2")
        if count_chunks(conn) == 0:
            pytest.skip("no chunks ingested")
    except Exception as exc:  # database down or schema missing
        pytest.skip(f"database not available: {exc}")
    yield Retriever(conn, OllamaEmbedder(settings.ollama_host, settings.embedding_model))
    conn.close()


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
