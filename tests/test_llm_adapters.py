import json

import httpx

from energy_agent.llm import Message, ToolCall
from energy_agent.llm.mistral import MistralChat
from energy_agent.llm.ollama import OllamaChat

CONVERSATION = [
    Message("user", "Status LW-2026-0003?"),
    Message(
        "assistant",
        tool_calls=(ToolCall("abc123XYZ", "get_switch_status", {"request_id": "LW-2026-0003"}),),
    ),
    Message("tool", '{"found": 1}', tool_call_id="abc123XYZ", name="get_switch_status"),
]
TOOLS = [{"type": "function", "function": {"name": "get_switch_status", "parameters": {}}}]


def mock_client(base_url: str, responses: list[httpx.Response], sent: list[dict]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return responses.pop(0)

    return httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler))


def test_mistral_wire_format_and_tool_call_parsing():
    sent: list[dict] = []
    reply = {
        "model": "ministral-8b-latest",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "x1",
                            "type": "function",
                            "function": {
                                "name": "calculate_deadline",
                                "arguments": '{"process": "lieferbeginn", "step": 5}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 87, "completion_tokens": 24},
    }
    client = mock_client("https://api.mistral.ai", [httpx.Response(200, json=reply)], sent)
    chat = MistralChat("key", "ministral-8b-latest", client=client)

    response = chat.chat(CONVERSATION, TOOLS)

    assert response.message.tool_calls == (
        ToolCall("x1", "calculate_deadline", {"process": "lieferbeginn", "step": 5}),
    )
    assert (response.prompt_tokens, response.completion_tokens) == (87, 24)
    payload = sent[0]
    assert payload["tool_choice"] == "auto" and payload["temperature"] == 0
    assistant, tool = payload["messages"][1], payload["messages"][2]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"request_id": "LW-2026-0003"}'
    assert tool == {
        "role": "tool",
        "content": '{"found": 1}',
        "tool_call_id": "abc123XYZ",
        "name": "get_switch_status",
    }


def test_mistral_retries_on_rate_limit_using_retry_after():
    sent: list[dict] = []
    waits: list[float] = []
    ok = {"choices": [{"message": {"content": [{"type": "text", "text": "Hallo"}]}}]}
    client = mock_client(
        "https://api.mistral.ai",
        [httpx.Response(429, headers={"retry-after": "1.5"}), httpx.Response(200, json=ok)],
        sent,
    )
    chat = MistralChat("key", "m", client=client, sleep=waits.append)

    response = chat.chat([Message("user", "Hi")], [])

    assert response.message.content == "Hallo"  # content given as typed chunks
    assert waits == [1.5]
    assert len(sent) == 2 and "tools" not in sent[0]


def test_ollama_wire_format_and_generated_call_ids():
    sent: list[dict] = []
    reply = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search_documents", "arguments": {"query": "Ersatz"}}}
            ],
        },
        "prompt_eval_count": 500,
        "eval_count": 20,
    }
    client = mock_client("http://ollama", [httpx.Response(200, json=reply)], sent)
    chat = OllamaChat("http://ollama", "qwen3:8b", client=client)

    response = chat.chat(CONVERSATION, TOOLS)

    call = response.message.tool_calls[0]
    assert call.name == "search_documents" and call.arguments == {"query": "Ersatz"}
    assert call.id.startswith("call_")
    assert (response.prompt_tokens, response.completion_tokens) == (500, 20)
    payload = sent[0]
    assert payload["stream"] is False and payload["think"] is False
    assert payload["options"]["num_ctx"] == 16384
    assert payload["messages"][1]["tool_calls"][0]["function"]["arguments"] == {
        "request_id": "LW-2026-0003"
    }
    assert payload["messages"][2] == {
        "role": "tool",
        "content": '{"found": 1}',
        "tool_name": "get_switch_status",
    }
