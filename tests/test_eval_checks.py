from datetime import date

from energy_agent.agent import AgentResult
from energy_agent.evaluation.checks import (
    evaluate,
    find_dates,
    find_times,
    has_citation,
    mentions_date,
    wrong_weekdays,
)
from energy_agent.evaluation.golden import GoldenCase


def test_find_dates_in_german_and_iso_formats():
    text = "Am 6.6.2025, am 11. Juni 2025 und am 2026-12-28T11:00; Version 1.2.3 ist kein Datum."
    assert [d for _, d in find_dates(text)] == [
        date(2025, 6, 6),
        date(2025, 6, 11),
        date(2026, 12, 28),
    ]


def test_find_times_variants():
    assert find_times("um 11:00 Uhr, 5.00 Uhr, 6 Uhr und 2026-12-28T07:30") == {
        "11:00",
        "05:00",
        "06:00",
        "07:30",
    }
    assert find_times("S. 21, Nr. 6, Abschnitt 2.1.2") == set()


def test_mentions_date_accepts_short_form_but_not_wrong_year():
    assert mentions_date("bis Mo 28.12. um 11 Uhr", date(2026, 12, 28))
    assert not mentions_date("bis 28.12.2025", date(2026, 12, 28))


def test_wrong_weekday_is_detected():
    assert wrong_weekdays("frühestens am Montag, 11. Juni 2025") == ["Montag 11.06.2025"]
    assert wrong_weekdays("frühestens am Mittwoch, 11. Juni 2025") == []
    assert wrong_weekdays("Di 27.10.2026, 11:00 Uhr") == []


def test_citation_needs_document_and_locator():
    assert has_citation("Quelle: GPKE Teil 2, 2.1.2 SD: Lieferbeginn, Nr. 6, S. 21")
    assert not has_citation("laut GPKE")
    assert not has_citation("siehe S. 21")


CASE = GoldenCase(
    id="t",
    category="deadline",
    question="Anmeldung am 23.12.2026 um 10:00 Uhr. Bis wann Ablehnung?",
    tools=["calculate_deadline"],
    deadline_args={"process": "lieferbeginn", "step": [6]},
    dates=[date(2026, 12, 28)],
    times=["11:00"],
)
TOOL_CALL = {
    "name": "calculate_deadline",
    "arguments": {"process": "lieferbeginn", "step": "6", "step1_sent_at": "2026-12-23T10:00"},
    "result": {"deadline": "2026-12-28T11:00", "source": "GPKE Teil 2 ..."},
}


def agent_result(answer: str, tool_calls: list[dict]) -> AgentResult:
    return AgentResult("q", answer, "run", "answer", tool_calls=tool_calls)


def test_correct_grounded_answer_passes():
    answer = (
        "Spätestens Mo 28.12.2026, 11:00 Uhr (GPKE Teil 2, 2.1.2 SD: Lieferbeginn, Nr. 6, S. 21)."
    )
    result = evaluate(CASE, agent_result(answer, [TOOL_CALL]))
    assert result.passed and result.cites and result.grounded, result.failures


def test_answer_restating_the_tool_result_wrongly_is_not_faithful():
    answer = "Spätestens Mittwoch, 29.12.2026, 12:00 Uhr."  # 29.12.2026 is a Tuesday
    result = evaluate(CASE, agent_result(answer, [TOOL_CALL]))
    assert not result.passed and not result.faithful
    assert any("29.12.2026" in f for f in result.failures)
    assert any("12:00" in f for f in result.failures)
    assert any("wrong weekday" in f for f in result.failures)


def test_missing_tool_and_wrong_step_are_reported():
    wrong_step = {**TOOL_CALL, "arguments": {"process": "lieferbeginn", "step": 2}}
    result = evaluate(CASE, agent_result("28.12.2026, 11:00 Uhr", [wrong_step]))
    assert result.tools_ok and not result.args_ok and not result.passed
    no_tools = evaluate(CASE, agent_result("28.12.2026, 11:00 Uhr", []))
    assert not no_tools.tools_ok and not no_tools.faithful  # date not backed by any tool
