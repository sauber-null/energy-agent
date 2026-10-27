"""Run the golden set through an agent and summarise the results."""

import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from energy_agent.agent import Agent
from energy_agent.evaluation.checks import CaseResult, evaluate
from energy_agent.evaluation.golden import GoldenCase
from energy_agent.sources import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def run_golden(
    agent: Agent,
    cases: list[GoldenCase],
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    results = []
    for case in cases:
        result = evaluate(case, agent.run(case.question))
        results.append(result)
        if on_result:
            on_result(result)
    return results


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[CaseResult]) -> dict:
    categories = sorted({r.category for r in results})
    return {
        "cases": len(results),
        "pass_rate": _rate([r.passed for r in results]),
        "tool_use": _rate([r.tools_ok for r in results]),
        "deadline_args": _rate([r.args_ok for r in results]),
        "answer_correct": _rate([r.answer_ok for r in results]),
        "faithful": _rate([r.faithful for r in results]),
        "cites_source": _rate([r.cites for r in results if r.cites is not None]),
        "grounded": _rate([r.grounded for r in results]),
        "guardrail_fired": sum(r.guardrail_retries > 0 for r in results),
        "by_category": {
            c: _rate([r.passed for r in results if r.category == c]) for c in categories
        },
        "avg_latency_s": round(sum(r.latency_ms for r in results) / len(results) / 1000, 1),
        "avg_llm_calls": round(sum(r.llm_calls for r in results) / len(results), 2),
        "tokens": sum(r.prompt_tokens + r.completion_tokens for r in results),
    }


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    return out.stdout.strip()


def save_results(results: list[CaseResult], summary: dict, meta: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{meta['provider']}_{meta['model']}_{'guard' if meta['guardrail'] else 'noguard'}"
    path = RESULTS_DIR / f"{name.replace(':', '-').replace('/', '-')}.json"
    payload = {
        "meta": {
            **meta,
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "commit": _git_commit(),
        },
        "summary": summary,
        "cases": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return path
