from energy_agent.config import Settings
from energy_agent.llm.base import ChatModel, ChatResponse, Message, ToolCall
from energy_agent.llm.mistral import MistralChat
from energy_agent.llm.ollama import OllamaChat

__all__ = ["ChatModel", "ChatResponse", "Message", "ToolCall", "create_chat_model"]


def create_chat_model(settings: Settings, provider: str | None = None) -> ChatModel:
    provider = provider or settings.llm_provider
    if provider == "mistral":
        if settings.mistral_api_key is None:
            raise ValueError("MISTRAL_API_KEY is not set; add it to .env")
        return MistralChat(settings.mistral_api_key.get_secret_value(), settings.mistral_model)
    if provider == "ollama":
        return OllamaChat(settings.ollama_host, settings.ollama_model)
    raise ValueError(f"unknown provider {provider!r}")
