"""Command line entry point: `energy-agent <command>`."""

import argparse
import json
from dataclasses import asdict

from energy_agent.config import get_settings


def cmd_info(args: argparse.Namespace) -> None:
    settings = get_settings()
    print(f"provider={settings.llm_provider}, model={settings.active_model}")


def cmd_chunk(args: argparse.Namespace) -> None:
    from energy_agent.rag.chunking import chunk_document
    from energy_agent.rag.pdf import pdf_to_markdown_pages
    from energy_agent.sources import PROCESSED_DIR, load_sources

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "chunks.jsonl"
    with out.open("w") as f:
        for source in load_sources():
            chunks = chunk_document(source.id, pdf_to_markdown_pages(source.path), args.max_chars)
            for chunk in chunks:
                f.write(json.dumps({"id": chunk.id, **asdict(chunk)}, ensure_ascii=False) + "\n")
            sizes = [len(c.text) for c in chunks]
            print(
                f"{source.id}: {len(chunks)} chunks, "
                f"avg {sum(sizes) // len(sizes)} chars, max {max(sizes)}"
            )
    print(f"written to {out}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from energy_agent.rag.embeddings import OllamaEmbedder
    from energy_agent.rag.ingest import ingest
    from energy_agent.rag.store import connect, count_chunks
    from energy_agent.sources import load_sources

    settings = get_settings()
    embedder = OllamaEmbedder(settings.ollama_host, settings.embedding_model)
    with connect(settings.database_url) as conn:
        ingest(conn, embedder, load_sources(), args.max_chars)
        print(f"{count_chunks(conn)} chunks in the database")


def cmd_search(args: argparse.Namespace) -> None:
    from energy_agent.rag.embeddings import OllamaEmbedder
    from energy_agent.rag.search import Retriever
    from energy_agent.rag.store import connect

    settings = get_settings()
    embedder = OllamaEmbedder(settings.ollama_host, settings.embedding_model)
    with connect(settings.database_url) as conn:
        results = Retriever(conn, embedder).search(args.query, args.k, args.mode)
    for i, r in enumerate(results, start=1):
        pages = f"S. {r.page_start}" + (f"-{r.page_end}" if r.page_end != r.page_start else "")
        print(f"\n[{i}] {r.score:.4f}  {r.doc_id}, {pages}\n    {r.section}")
        print("    " + r.content[:300].replace("\n", "\n    "))


def cmd_seed(args: argparse.Namespace) -> None:
    from energy_agent.rag.store import connect
    from energy_agent.tools.switch_requests import seed

    with connect(get_settings().database_url) as conn:
        print(f"{seed(conn)} switch requests in the test database")


def cmd_tool(args: argparse.Namespace) -> None:
    from energy_agent.rag.embeddings import OllamaEmbedder
    from energy_agent.rag.search import Retriever
    from energy_agent.rag.store import connect
    from energy_agent.tools.registry import build_tools

    settings = get_settings()
    embedder = OllamaEmbedder(settings.ollama_host, settings.embedding_model)
    with connect(settings.database_url) as conn:
        tools = {t.name: t for t in build_tools(conn, Retriever(conn, embedder))}
        if args.name not in tools:
            raise SystemExit(f"unknown tool {args.name!r}, available: {', '.join(tools)}")
        result = tools[args.name].run(args.arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _print_event(event: dict) -> None:
    if event["event"] == "tool_call":
        args = json.dumps(event["arguments"], ensure_ascii=False)
        status = f"error: {event['error']}" if event["error"] else "ok"
        print(f"  → {event['tool']}({args})  [{status}]")
    elif event["event"] == "llm_call":
        print(f"  · LLM call {event['step']}: {event['latency_ms']} ms")


def cmd_ask(args: argparse.Namespace) -> None:
    from energy_agent.agent import Agent
    from energy_agent.llm import create_chat_model
    from energy_agent.rag.embeddings import OllamaEmbedder
    from energy_agent.rag.search import Retriever
    from energy_agent.rag.store import connect
    from energy_agent.tools.registry import build_tools
    from energy_agent.tracing import TRACE_FILE, Tracer

    settings = get_settings()
    model = create_chat_model(settings, args.provider)
    embedder = OllamaEmbedder(settings.ollama_host, settings.embedding_model)
    print(f"{model.provider} / {model.model}")
    with connect(settings.database_url) as conn:
        tools = build_tools(conn, Retriever(conn, embedder))
        agent = Agent(model, tools, Tracer(on_event=_print_event))
        result = agent.run(args.question)
    print(f"\n{result.answer}\n")
    print(
        f"[{result.llm_calls} LLM calls, {len(result.tool_calls)} tool calls, "
        f"{result.prompt_tokens}+{result.completion_tokens} tokens, "
        f"{result.latency_ms / 1000:.1f} s, trace {result.run_id} in {TRACE_FILE.name}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="energy-agent")
    commands = parser.add_subparsers(required=True)

    commands.add_parser("info", help="show the active LLM provider and model").set_defaults(
        func=cmd_info
    )

    chunk = commands.add_parser("chunk", help="parse the PDFs into data/processed/chunks.jsonl")
    chunk.add_argument("--max-chars", type=int, default=1500)
    chunk.set_defaults(func=cmd_chunk)

    ingest = commands.add_parser("ingest", help="parse, embed and store all source documents")
    ingest.add_argument("--max-chars", type=int, default=1500)
    ingest.set_defaults(func=cmd_ingest)

    search = commands.add_parser("search", help="search the document chunks")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--mode", choices=["hybrid", "vector", "text"], default="hybrid")
    search.set_defaults(func=cmd_search)

    commands.add_parser("seed", help="(re)create the switch request test data").set_defaults(
        func=cmd_seed
    )

    tool = commands.add_parser("tool", help="call one agent tool directly with JSON arguments")
    tool.add_argument("name")
    tool.add_argument("arguments", help='JSON, e.g. \'{"request_id": "LW-2026-0003"}\'')
    tool.set_defaults(func=cmd_tool)

    ask = commands.add_parser("ask", help="ask the agent a question")
    ask.add_argument("question")
    ask.add_argument("--provider", choices=["mistral", "ollama"], help="default: LLM_PROVIDER")
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)
