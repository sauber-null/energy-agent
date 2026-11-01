"""Web demo: one page that streams every agent step while the question is answered.

Meant to be reached through a tunnel, so it protects itself: HTTP Basic auth on every page
and API route, a global rate limit (protects the Mistral quota and the laptop), one question
at a time, a question length limit and noindex headers. The agent's tools only read data,
so a malicious question cannot change anything.
"""

import json
import logging
import queue
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from energy_agent.agent import AgentResult
from energy_agent.config import Settings

log = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).parent / "static" / "index.html"
MAX_RESULT_CHARS = 1500  # tool results shown in the page are cut to keep the stream small

Provider = Literal["mistral", "ollama"]
# run_agent(question, provider, on_event) -> AgentResult
AgentRunner = Callable[[str, Provider, Callable[[dict], None]], AgentResult]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    provider: Provider = "mistral"


class RateLimiter:
    """Global sliding-window limits; the demo has one user at a time, not per-IP quotas."""

    def __init__(self, per_minute: int, per_day: int, clock: Callable[[], float] = time.time):
        self.limits = ((60, per_minute, "Minute"), (86_400, per_day, "Tag"))
        self.clock = clock
        self.calls: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            now = self.clock()
            while self.calls and now - self.calls[0] > 86_400:
                self.calls.popleft()
            # Messages are shown on the (German) demo page as they are.
            for window, limit, unit in self.limits:
                if sum(1 for t in self.calls if now - t <= window) >= limit:
                    raise HTTPException(
                        429, f"Limit erreicht: höchstens {limit} Fragen pro {unit}."
                    )
            self.calls.append(now)


def public_event(event: dict) -> dict | None:
    """The part of a trace event that the page shows."""
    kind = event["event"]
    if kind == "run_start":
        return {"event": kind, "provider": event["provider"], "model": event["model"]}
    if kind == "llm_call":
        keys = ("step", "latency_ms", "prompt_tokens", "completion_tokens", "tool_calls")
        return {"event": kind, **{k: event[k] for k in keys}}
    if kind == "tool_call":
        result = json.dumps(event["result"], ensure_ascii=False, default=str)
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + " …"
        return {
            "event": kind,
            "step": event["step"],
            "tool": event["tool"],
            "arguments": event["arguments"],
            "error": event["error"],
            "result": result,
        }
    if kind == "guardrail":
        return {"event": kind, "step": event["step"]}
    return None  # run_end: the final "answer" event carries the totals


def create_app(settings: Settings, run_agent: AgentRunner) -> FastAPI:
    if settings.demo_password is None or not settings.demo_password.get_secret_value():
        raise ValueError("DEMO_PASSWORD is not set; the demo must not run without a password")
    password = settings.demo_password.get_secret_value().encode()
    limiter = RateLimiter(settings.demo_questions_per_minute, settings.demo_questions_per_day)
    busy = threading.Lock()  # local models answer one question at a time
    basic = HTTPBasic(realm="energy-agent demo")

    def require_password(credentials: Annotated[HTTPBasicCredentials, Depends(basic)]) -> None:
        if not secrets.compare_digest(credentials.password.encode(), password):
            raise HTTPException(401, "Falsches Passwort", headers={"WWW-Authenticate": "Basic"})

    app = FastAPI(title="energy-agent demo", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def no_index(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response

    @app.get("/robots.txt", response_class=PlainTextResponse)
    def robots() -> str:
        return "User-agent: *\nDisallow: /\n"

    @app.get("/", dependencies=[Depends(require_password)])
    def index() -> FileResponse:
        return FileResponse(INDEX_HTML)

    @app.post("/api/ask", dependencies=[Depends(require_password)])
    def ask(request: AskRequest) -> StreamingResponse:
        if not busy.acquire(blocking=False):
            raise HTTPException(
                429, "Gerade wird eine andere Frage beantwortet, bitte kurz warten."
            )
        try:
            limiter.check()
        except HTTPException:
            busy.release()
            raise

        events: queue.Queue[dict | None] = queue.Queue()

        def on_event(event: dict) -> None:
            if (public := public_event(event)) is not None:
                events.put(public)

        def work() -> None:
            try:
                result = run_agent(request.question, request.provider, on_event)
                events.put(
                    {
                        "event": "answer",
                        "answer": result.answer,
                        "grounded": result.grounded,
                        "llm_calls": result.llm_calls,
                        "tool_calls": len(result.tool_calls),
                        "tokens": result.prompt_tokens + result.completion_tokens,
                        "latency_ms": result.latency_ms,
                        "run_id": result.run_id,
                    }
                )
            except Exception:
                log.exception("agent run failed")
                events.put(
                    {"event": "error", "message": "Der Agent ist auf einen Fehler gestoßen."}
                )
            finally:
                events.put(None)
                busy.release()

        # The agent runs in its own thread so the stream can forward events as they happen;
        # the lock is released there, even if the browser disconnects.
        threading.Thread(target=work, daemon=True).start()

        def stream() -> Iterator[str]:
            while (item := events.get()) is not None:
                yield f"data: {json.dumps(item, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def default_runner(settings: Settings) -> AgentRunner:
    """Run the real agent: one database connection per question."""
    from energy_agent.agent import Agent
    from energy_agent.llm import create_chat_model
    from energy_agent.rag.embeddings import OllamaEmbedder
    from energy_agent.rag.search import Retriever
    from energy_agent.rag.store import connect
    from energy_agent.tools.registry import build_tools
    from energy_agent.tracing import Tracer

    embedder = OllamaEmbedder(settings.ollama_host, settings.embedding_model)
    models = {provider: create_chat_model(settings, provider) for provider in ("mistral", "ollama")}

    def run(question: str, provider: Provider, on_event: Callable[[dict], None]) -> AgentResult:
        with connect(settings.database_url) as conn:
            tools = build_tools(conn, Retriever(conn, embedder))
            return Agent(models[provider], tools, Tracer(on_event=on_event)).run(question)

    return run
