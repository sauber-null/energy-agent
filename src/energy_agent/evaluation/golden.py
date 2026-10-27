"""The golden question set (eval/golden.yaml)."""

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from energy_agent.sources import PROJECT_ROOT

GOLDEN_FILE = PROJECT_ROOT / "eval" / "golden.yaml"
EVAL_TODAY = date(2026, 10, 27)  # the switch request test data is relative to this day

Category = Literal["deadline", "status", "combined", "rag", "negative"]


class DeadlineArgs(BaseModel):
    process: str
    step: list[int]


class GoldenCase(BaseModel):
    id: str
    category: Category
    question: str
    tools: list[str] = []
    deadline_args: DeadlineArgs | None = None
    dates: list[date] = []
    times: list[str] = []
    keywords: list[list[str]] = []
    forbidden: list[str] = []
    section: list[str] = []
    reference_call: dict | None = None

    @property
    def needs_citation(self) -> bool:
        """Answers stating a rule or deadline should name their source."""
        return self.category in ("deadline", "rag")


def load_golden(path: Path = GOLDEN_FILE) -> list[GoldenCase]:
    return [GoldenCase.model_validate(case) for case in yaml.safe_load(path.read_text())]
