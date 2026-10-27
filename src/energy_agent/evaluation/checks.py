"""Deterministic checks of an agent answer against the golden case and the tool results.

No LLM judge: every check is a regular expression or set comparison, so results are
reproducible and cheap. "Faithful" means the answer does not contain dates or times that
appear neither in the question nor in any tool result, and that every weekday written next
to a date is the right one. This catches the typical failure of small models: the tool
returns the correct deadline, the model restates it wrongly.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date

from energy_agent.agent import AgentResult
from energy_agent.evaluation.golden import GoldenCase

MONTHS = {
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4, "mai": 5, "juni": 6,
    "juli": 7, "august": 8, "september": 9, "oktober": 10, "november": 11, "dezember": 12,
}  # fmt: skip
WEEKDAYS = {
    "montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3, "freitag": 4, "samstag": 5,
    "sonntag": 6, "mo": 0, "di": 1, "mi": 2, "do": 3, "fr": 4, "sa": 5, "so": 6,
}  # fmt: skip

_NUMERIC_DATE = re.compile(r"(?<![\d.])(\d{1,2})\.\s?(\d{1,2})\.\s?(\d{4})(?!\d)")
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_NAMED_DATE = re.compile(
    rf"(?<!\d)(\d{{1,2}})\.?\s+({'|'.join(MONTHS)})\s+(\d{{4}})(?!\d)", re.IGNORECASE
)
_TIMES = (
    re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?!\d)"),  # 11:00, also in ISO datetimes
    re.compile(r"(?<![\d.:])([01]?\d|2[0-3])\.([0-5]\d)(?=\s*Uhr)"),  # 11.00 Uhr
    re.compile(r"(?<![\d.:])([01]?\d|2[0-3])()\s*Uhr"),  # 11 Uhr
)
_WEEKDAY = re.compile(rf"\b({'|'.join(WEEKDAYS)})\b\.?,?\s+(?:den\s+)?", re.IGNORECASE)
_SOURCE_DOC = re.compile(r"GPKE|BK6-22-024|Beschluss|Festlegung")
_SOURCE_LOCATOR = re.compile(r"S\.\s?\d+|Seite\s+\d+|Nr\.\s?\d+|\b\d+\.\d+(?:\.\d+)+")


def find_dates(text: str) -> list[tuple[int, date]]:
    """(position, date) for every full date in the text, in any of the usual formats."""
    found = []
    for match in _NUMERIC_DATE.finditer(text):
        day, month, year = map(int, match.groups())
        found.append((match.start(), day, month, year))
    for match in _ISO_DATE.finditer(text):
        year, month, day = map(int, match.groups())
        found.append((match.start(), day, month, year))
    for match in _NAMED_DATE.finditer(text):
        day, month, year = int(match[1]), MONTHS[match[2].lower()], int(match[3])
        found.append((match.start(), day, month, year))
    dates = []
    for pos, day, month, year in found:
        try:
            dates.append((pos, date(year, month, day)))
        except ValueError:
            continue  # e.g. a version number that looks like a date
    return sorted(dates)


def find_times(text: str) -> set[str]:
    return {
        f"{int(match[1]):02d}:{match[2] or '00'}"
        for pattern in _TIMES
        for match in pattern.finditer(text)
    }


def mentions_date(text: str, expected: date) -> bool:
    if expected in {d for _, d in find_dates(text)}:
        return True
    # Also accept "28.12." without the year.
    short = rf"(?<![\d.])0?{expected.day}\.\s?0?{expected.month}\.(?!\s?\d)"
    return re.search(short, text) is not None


def wrong_weekdays(text: str) -> list[str]:
    """Weekday names that do not match the date right after them ('Montag, 11.06.2025')."""
    dates = find_dates(text)
    wrong = []
    for match in _WEEKDAY.finditer(text):
        following = next((d for pos, d in dates if match.end() <= pos <= match.end() + 2), None)
        if following and following.weekday() != WEEKDAYS[match[1].lower()]:
            wrong.append(f"{match[1]} {following:%d.%m.%Y}")
    return wrong


def has_citation(text: str) -> bool:
    return bool(_SOURCE_DOC.search(text) and _SOURCE_LOCATOR.search(text))


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    tools_ok: bool
    args_ok: bool
    answer_ok: bool
    faithful: bool
    cites: bool | None
    grounded: bool
    guardrail_retries: int = 0
    failures: list[str] = field(default_factory=list)
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _step_matches(arguments: dict | str, case: GoldenCase) -> bool:
    if not isinstance(arguments, dict) or arguments.get("process") != case.deadline_args.process:
        return False
    try:
        return int(arguments.get("step")) in case.deadline_args.step
    except (TypeError, ValueError):
        return False


def evaluate(case: GoldenCase, result: AgentResult) -> CaseResult:
    answer = result.answer or ""
    lower = answer.lower()
    failures: list[str] = []

    called = {call["name"] for call in result.tool_calls}
    missing = [tool for tool in case.tools if tool not in called]
    tools_ok = not missing
    if missing:
        failures.append(f"tool not called: {', '.join(missing)}")

    args_ok = True
    if case.deadline_args:
        args_ok = any(
            call["name"] == "calculate_deadline" and _step_matches(call["arguments"], case)
            for call in result.tool_calls
        )
        if not args_ok:
            used = [c["arguments"] for c in result.tool_calls if c["name"] == "calculate_deadline"]
            failures.append(f"wrong deadline args {used}, expected {case.deadline_args}")

    answer_failures = [
        f"date {d:%d.%m.%Y} missing" for d in case.dates if not mentions_date(answer, d)
    ]
    answer_failures += [f"time {t} missing" for t in case.times if t not in find_times(answer)]
    answer_failures += [
        f"none of {group} in answer"
        for group in case.keywords
        if not any(word.lower() in lower for word in group)
    ]
    answer_failures += [
        f"forbidden '{word}' in answer" for word in case.forbidden if word.lower() in lower
    ]
    answer_ok = not answer_failures
    failures += answer_failures

    evidence = case.question + json.dumps(
        [call["result"] for call in result.tool_calls], ensure_ascii=False, default=str
    )
    evidence_dates = {d for _, d in find_dates(evidence)}
    invented_dates = sorted({d for _, d in find_dates(answer)} - evidence_dates)
    invented_times = sorted(find_times(answer) - find_times(evidence) - {"00:00"})
    weekdays = wrong_weekdays(answer)
    faithful = not (invented_dates or invented_times or weekdays)
    if invented_dates:
        failures.append(f"dates not in tool results: {[f'{d:%d.%m.%Y}' for d in invented_dates]}")
    if invented_times:
        failures.append(f"times not in tool results: {invented_times}")
    if weekdays:
        failures.append(f"wrong weekday: {weekdays}")

    return CaseResult(
        id=case.id,
        category=case.category,
        passed=tools_ok and args_ok and answer_ok and faithful,
        tools_ok=tools_ok,
        args_ok=args_ok,
        answer_ok=answer_ok,
        faithful=faithful,
        cites=has_citation(answer) if case.needs_citation else None,
        grounded=result.grounded,
        guardrail_retries=result.guardrail_retries,
        failures=failures,
        answer=answer,
        tool_calls=[{"name": c["name"], "arguments": c["arguments"]} for c in result.tool_calls],
        llm_calls=result.llm_calls,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
