"""Text embeddings from a local Ollama model (default bge-m3: multilingual, 1024 dimensions)."""

import httpx

EMBEDDING_DIM = 1024


class OllamaEmbedder:
    def __init__(
        self,
        host: str,
        model: str,
        batch_size: int = 16,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self._client = client or httpx.Client(base_url=host, timeout=120)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._client.post("/api/embed", json={"model": self.model, "input": batch})
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
        return vectors
