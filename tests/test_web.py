import json
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from energy_agent.agent import AgentResult
from energy_agent.config import Settings
from energy_agent.web.app import RateLimiter, create_app

AUTH = ("demo", "test-password")


def fake_runner(question, provider, on_event):
    on_event({"event": "run_start", "provider": provider, "model": "fake-1", "question": question})
    on_event(
        {
            "event": "llm_call",
            "step": 1,
            "latency_ms": 5,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "tool_calls": ["calculate_deadline"],
            "content": "",
        }
    )
    on_event(
        {
            "event": "tool_call",
            "step": 1,
            "tool": "calculate_deadline",
            "arguments": {"process": "lieferbeginn", "step": 6},
            "error": None,
            "result": {"deadline": "2026-12-28T11:00", "text": "x" * 5000},
        }
    )
    tool_calls = [{"name": "calculate_deadline", "arguments": {}, "result": {"deadline": "x"}}]
    return AgentResult(question, "Bis **28.12.2026**", "run1", "answer", 2, tool_calls, 20, 5, 10)


def make_client(runner=fake_runner, **overrides) -> TestClient:
    settings = Settings(_env_file=None, demo_password=AUTH[1], **overrides)
    return TestClient(create_app(settings, runner))


def read_events(response) -> list[dict]:
    return [
        json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")
    ]


def test_server_refuses_to_start_without_password():
    with pytest.raises(ValueError, match="DEMO_PASSWORD"):
        create_app(Settings(_env_file=None), fake_runner)


def test_pages_require_the_password():
    client = make_client()
    assert client.get("/").status_code == 401
    assert client.get("/", auth=("demo", "wrong")).status_code == 401
    assert client.post("/api/ask", json={"question": "Frist?"}).status_code == 401
    page = client.get("/", auth=AUTH)
    assert page.status_code == 200 and "Energy Agent" in page.text
    assert page.headers["X-Robots-Tag"] == "noindex, nofollow"
    assert client.get("/robots.txt").text.endswith("Disallow: /\n")
    assert client.get("/docs").status_code == 404  # no public API docs


def test_ask_streams_steps_then_answer():
    response = make_client().post(
        "/api/ask", json={"question": "Bis wann?", "provider": "ollama"}, auth=AUTH
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = read_events(response)
    assert [e["event"] for e in events] == ["run_start", "llm_call", "tool_call", "answer"]
    assert events[0]["provider"] == "ollama"
    assert len(events[2]["result"]) < 1600  # long tool results are cut for the page
    assert events[3]["answer"] == "Bis **28.12.2026**" and events[3]["grounded"]


def test_question_length_is_limited():
    response = make_client().post("/api/ask", json={"question": "x" * 501}, auth=AUTH)
    assert response.status_code == 422


def test_agent_failure_becomes_an_error_event():
    def broken(question, provider, on_event):
        raise RuntimeError("model down")

    events = read_events(
        make_client(broken).post("/api/ask", json={"question": "Frist?"}, auth=AUTH)
    )
    assert events == [{"event": "error", "message": "The agent failed, see server log."}]


def test_rate_limit_per_minute():
    client = make_client(demo_questions_per_minute=2)
    statuses = [
        client.post("/api/ask", json={"question": "Frist?"}, auth=AUTH).status_code
        for _ in range(3)
    ]
    assert statuses == [200, 200, 429]


def test_second_question_while_busy_is_rejected():
    started, release = threading.Event(), threading.Event()

    def slow(question, provider, on_event):
        started.set()
        release.wait(5)
        return fake_runner(question, provider, on_event)

    client = make_client(slow)
    first = threading.Thread(
        target=lambda: client.post("/api/ask", json={"question": "eins"}, auth=AUTH)
    )
    first.start()
    started.wait(5)
    busy = client.post("/api/ask", json={"question": "zwei"}, auth=AUTH)
    release.set()
    first.join(5)
    assert busy.status_code == 429 and "Another question" in busy.json()["detail"]


def test_rate_limiter_window_slides():
    now = [0.0]
    limiter = RateLimiter(per_minute=1, per_day=10, clock=lambda: now[0])
    limiter.check()
    with pytest.raises(HTTPException):
        limiter.check()
    now[0] = 61
    limiter.check()
