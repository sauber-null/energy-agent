"""Ollama adapter for local models (native /api/chat with tool support)."""

import time
import uuid
from collections.abc import Sequence

import httpx

from energy_agent.llm.base import ChatResponse, Message, ToolCall, parse_arguments


class OllamaChat:
    provider = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        client: httpx.Client | None = None,
        num_ctx: int = 16384,
        think: bool | None = False,
    ):
        self.model = model
        # Ollama's default context window is small and overflowing it truncates the prompt
        # silently, which would cut off retrieved passages.
        self.num_ctx = num_ctx
        # Reasoning models such as qwen3 "think" before answering unless told not to;
        # None leaves the model default for models without that switch.
        self.think = think
        self._client = client or httpx.Client(base_url=host, timeout=300)

    def chat(self, messages: Sequence[Message], tools: Sequence[dict]) -> ChatResponse:
        payload: dict = {
            "model": self.model,
            "messages": [_to_wire(m) for m in messages],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = list(tools)
        if self.think is not None:
            payload["think"] = self.think

        started = time.perf_counter()
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        latency_ms = round((time.perf_counter() - started) * 1000)

        wire = data["message"]
        tool_calls = tuple(
            # Ollama does not assign call ids; the agent needs one to pair results with calls.
            ToolCall(
                tc.get("id") or f"call_{uuid.uuid4().hex[:9]}",
                tc["function"]["name"],
                parse_arguments(tc["function"].get("arguments")),
            )
            for tc in wire.get("tool_calls") or []
        )
        return ChatResponse(
            message=Message("assistant", wire.get("content") or "", tool_calls),
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
        )


def _to_wire(message: Message) -> dict:
    wire: dict = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "function": {
                    "name": call.name,
                    "arguments": call.arguments if isinstance(call.arguments, dict) else {},
                }
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        wire["tool_name"] = message.name
    return wire
