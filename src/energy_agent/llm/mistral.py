"""Mistral La Plateforme adapter (chat completions with function calling)."""

import json
import time
from collections.abc import Callable, Sequence

import httpx

from energy_agent.llm.base import ChatResponse, Message, ToolCall, parse_arguments

API_URL = "https://api.mistral.ai"


class MistralChat:
    provider = "mistral"

    def __init__(
        self,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = client or httpx.Client(
            base_url=API_URL, timeout=120, headers={"Authorization": f"Bearer {api_key}"}
        )

    def chat(self, messages: Sequence[Message], tools: Sequence[dict]) -> ChatResponse:
        payload: dict = {
            "model": self.model,
            "messages": [_to_wire(m) for m in messages],
            "temperature": 0,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        started = time.perf_counter()
        data = self._post(payload)
        latency_ms = round((time.perf_counter() - started) * 1000)

        wire = data["choices"][0]["message"]
        tool_calls = tuple(
            ToolCall(tc["id"], tc["function"]["name"], parse_arguments(tc["function"]["arguments"]))
            for tc in wire.get("tool_calls") or []
        )
        usage = data.get("usage") or {}
        return ChatResponse(
            message=Message("assistant", _text(wire.get("content")), tool_calls),
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )

    def _post(self, payload: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            response = self._client.post("/v1/chat/completions", json=payload)
            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable and attempt < self.max_retries:
                # The free tier allows few requests per second; honour Retry-After if sent.
                self._sleep(float(response.headers.get("retry-after", 2**attempt)))
                continue
            response.raise_for_status()
            return response.json()
        raise AssertionError("unreachable")


def _text(content: str | list | None) -> str:
    # Some Mistral models return content as a list of typed chunks.
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _to_wire(message: Message) -> dict:
    wire: dict = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments
                    if isinstance(call.arguments, str)
                    else json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        wire["tool_call_id"] = message.tool_call_id
        wire["name"] = message.name
    return wire
