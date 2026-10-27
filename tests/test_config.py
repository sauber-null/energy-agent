import pytest
from pydantic import ValidationError

from energy_agent.config import Settings

ENV_VARS = ("LLM_PROVIDER", "MISTRAL_API_KEY", "MISTRAL_MODEL", "OLLAMA_HOST", "OLLAMA_MODEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "mistral"
    assert settings.mistral_api_key is None
    assert settings.active_model == settings.mistral_model


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.active_model == "llama3.1:8b"


def test_api_key_is_masked(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "secret-123")
    settings = Settings(_env_file=None)
    assert "secret-123" not in repr(settings)
    assert settings.mistral_api_key.get_secret_value() == "secret-123"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "foo")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
