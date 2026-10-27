"""The tools the agent can call, with JSON schemas for function calling.

Arguments are validated with pydantic. Invalid arguments and domain errors are returned to
the model as {"error": ...} instead of raised, so the model can correct its call.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

import psycopg
from pydantic import BaseModel, Field, ValidationError, model_validator

from energy_agent.rag.search import Retriever
from energy_agent.tools.deadlines import Process, calculate_deadline
from energy_agent.tools.switch_requests import get_switch_status


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[BaseModel], dict]

    def schema(self) -> dict:
        """OpenAI-style function schema, accepted by both Mistral and Ollama."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _strip_titles(self.args_model.model_json_schema()),
            },
        }

    def run(self, arguments: dict | str) -> dict:
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            return self.func(self.args_model.model_validate(arguments))
        except (ValidationError, ValueError) as exc:  # JSONDecodeError is a ValueError
            return {"error": str(exc)}


def _strip_titles(schema: dict) -> dict:
    """pydantic adds a "title" to every field; it only costs tokens in the prompt."""
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


class SearchDocumentsArgs(BaseModel):
    query: str = Field(description="Search query in German, e.g. 'Frist Ablehnung Anmeldung'")
    k: int = Field(5, ge=1, le=10, description="Number of passages to return")


class CalculateDeadlineArgs(BaseModel):
    process: Process = Field(
        description="GPKE process: malo_ermittlung, kuendigung, lieferbeginn, "
        "lieferende_lf (Abmeldung by the supplier), lieferende_nb (initiated by the grid "
        "operator), ersatzversorgung_beginn"
    )
    step: int = Field(description="Step number (Nr.) in the GPKE process table")
    step1_sent_at: datetime | None = Field(
        None, description="When the step 1 message was sent, ISO 8601, e.g. 2026-10-26T17:20"
    )
    assignment_date: date | None = Field(
        None,
        description="Zuordnungsbeginn (supply start) or Zuordnungsende (supply end), "
        "ISO date, e.g. 2026-11-02",
    )


class GetSwitchStatusArgs(BaseModel):
    request_id: str | None = Field(None, description="Request ID, e.g. LW-2026-0003")
    malo_id: str | None = Field(None, description="11-digit Marktlokations-ID")

    @model_validator(mode="after")
    def one_identifier(self) -> "GetSwitchStatusArgs":
        if not (self.request_id or self.malo_id):
            raise ValueError("request_id or malo_id is required")
        return self


def build_tools(conn: psycopg.Connection, retriever: Retriever) -> list[Tool]:
    def search_documents(args: SearchDocumentsArgs) -> dict:
        results = retriever.search(args.query, args.k)
        return {
            "passages": [
                {
                    "source": f"{r.doc_title}, {r.section}, S. {r.page_start}"
                    + (f"-{r.page_end}" if r.page_end != r.page_start else ""),
                    "text": r.content,
                }
                for r in results
            ]
        }

    return [
        Tool(
            "search_documents",
            "Search the official GPKE process descriptions and the BNetzA decision BK6-22-024 "
            "(supplier switch within 24 hours). Returns passages with their source. Use it "
            "for questions about roles, process steps, rules and definitions.",
            SearchDocumentsArgs,
            search_documents,
        ),
        Tool(
            "calculate_deadline",
            "Calculate the deadline (Frist) of a GPKE process step in German working days "
            "(Werktage, incl. nationwide counting of state holidays, 24.12. and 31.12.). For "
            "step 1 of lieferbeginn/lieferende_lf pass assignment_date to get the latest day "
            "to send, or step1_sent_at to get the earliest possible date. Other steps need "
            "step1_sent_at. Returns the deadline, an explanation and the source.",
            CalculateDeadlineArgs,
            lambda a: calculate_deadline(a.process, a.step, a.step1_sent_at, a.assignment_date),
        ),
        Tool(
            "get_switch_status",
            "Look up supplier switch requests (Lieferbeginn, Lieferende, Ersatzversorgung) in "
            "the test database by request ID or MaLo-ID. Returns status, dates, parties and "
            "the rejection reason if any.",
            GetSwitchStatusArgs,
            lambda a: get_switch_status(conn, a.request_id, a.malo_id),
        ),
    ]
