from datetime import date, datetime

import pytest

from energy_agent.evaluation.checks import find_times, mentions_date
from energy_agent.evaluation.golden import load_golden
from energy_agent.tools.deadlines import RULES, calculate_deadline
from energy_agent.tools.registry import build_tools

CASES = load_golden()
TOOL_NAMES = {t.name for t in build_tools(conn=None, retriever=None)}


def test_golden_set_size_and_unique_ids():
    assert 25 <= len(CASES) <= 35
    assert len({c.id for c in CASES}) == len(CASES)


def test_every_case_checks_something_in_the_answer():
    for case in CASES:
        assert case.dates or case.times or case.keywords, case.id
        assert set(case.tools) <= TOOL_NAMES, case.id
        if case.category == "rag":
            assert case.section, case.id


def test_expected_deadline_steps_exist():
    known = {(r.process, r.step) for r in RULES}
    for case in CASES:
        if case.deadline_args:
            for step in case.deadline_args.step:
                assert (case.deadline_args.process, step) in known, case.id


@pytest.mark.parametrize("case", [c for c in CASES if c.reference_call], ids=lambda c: c.id)
def test_reference_call_reproduces_expected_answer(case):
    """The hand-derived expectations and the deadline tool must agree."""
    args = dict(case.reference_call)
    if "step1_sent_at" in args:
        args["step1_sent_at"] = datetime.fromisoformat(args["step1_sent_at"])
    if "assignment_date" in args:
        args["assignment_date"] = date.fromisoformat(args["assignment_date"])
    result = calculate_deadline(**args)
    rendered = " ".join(str(v) for v in result.values())
    for expected in case.dates:
        assert mentions_date(rendered, expected), (case.id, expected, result)
    assert set(case.times) <= find_times(rendered), (case.id, result)
