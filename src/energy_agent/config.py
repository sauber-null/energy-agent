"""Runtime configuration, read from environment variables and an optional .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Which chat model backend the agent talks to.
    llm_provider: Literal["mistral", "ollama"] = "mistral"

    # SecretStr keeps the key out of repr() and therefore out of logs and traces.
    mistral_api_key: SecretStr | None = None
    mistral_model: str = "mistral-small-latest"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"

    # Embeddings always run locally, independent of the chat provider.
    embedding_model: str = "bge-m3"

    database_url: str = "postgresql://energy:energy@localhost:5432/energy_agent"

    @property
    def active_model(self) -> str:
        return self.mistral_model if self.llm_provider == "mistral" else self.ollama_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
