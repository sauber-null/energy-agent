import pytest

from energy_agent.config import get_settings


@pytest.fixture(scope="session")
def db_conn():
    """Connection to the docker compose database; integration tests skip without it."""
    from energy_agent.rag.store import connect

    try:
        conn = connect(get_settings().database_url + "?connect_timeout=2")
    except Exception as exc:
        pytest.skip(f"database not available: {exc}")
    yield conn
    conn.close()
