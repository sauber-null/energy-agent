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


def main() -> None:
    parser = argparse.ArgumentParser(prog="energy-agent")
    commands = parser.add_subparsers(required=True)

    commands.add_parser("info", help="show the active LLM provider and model").set_defaults(
        func=cmd_info
    )

    chunk = commands.add_parser("chunk", help="parse the PDFs into data/processed/chunks.jsonl")
    chunk.add_argument("--max-chars", type=int, default=1500)
    chunk.set_defaults(func=cmd_chunk)

    args = parser.parse_args()
    args.func(args)
