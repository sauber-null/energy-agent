import json

import httpx

from energy_agent.rag.embeddings import OllamaEmbedder


def test_embed_sends_batches_and_keeps_order():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"embeddings": [[float(len(t))] for t in body["input"]]})

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    embedder = OllamaEmbedder("http://ollama", "bge-m3", batch_size=2, client=client)

    vectors = embedder.embed(["a", "bb", "ccc"])

    assert vectors == [[1.0], [2.0], [3.0]]
    assert [r["input"] for r in requests] == [["a", "bb"], ["ccc"]]
    assert all(r["model"] == "bge-m3" for r in requests)
