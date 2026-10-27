"""Provider-agnostic chat types.

The agent only knows these types and the ChatModel protocol. Each provider adapter
translates them to and from its own wire format, so switching between a hosted model
and a local one is a configuration change.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    # A dict when the model produced valid JSON, otherwise the raw string, which the
    # tool wrapper turns into an error message for the model.
    arguments: dict | str


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None  # role "tool": the call this result answers
    name: str | None = None  # role "tool": the tool that produced the result


@dataclass(frozen=True)
class ChatResponse:
    message: Message
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class ChatModel(Protocol):
    provider: str
    model: str

    def chat(self, messages: Sequence[Message], tools: Sequence[dict]) -> ChatResponse: ...


def parse_arguments(raw: dict | str | None) -> dict | str:
    if raw is None or isinstance(raw, dict):
        return raw or {}
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
    return parsed if isinstance(parsed, dict) else raw
