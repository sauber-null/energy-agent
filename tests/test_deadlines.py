from datetime import date, datetime

import pytest

from energy_agent.tools.deadlines import RULES, calculate_deadline


def test_registration_on_monday_allows_supply_from_wednesday():
    result = calculate_deadline("lieferbeginn", 1, step1_sent_at=datetime(2026, 10, 26, 17, 20))
    assert result["earliest_start"] == "2026-10-28"


def test_registration_before_pentecost_skips_weekend_and_holiday():
    result = calculate_deadline("lieferbeginn", 1, step1_sent_at=datetime(2025, 6, 6, 12, 0))
    assert result["earliest_start"] == "2025-06-11"
    assert "Pfingstmontag" in result["explanation"]


def test_latest_registration_day_for_a_given_supply_start():
    # Zuordnungsbeginn Mo 02.11.2026 -> last WT before is Fr 30.10. -> latest ÜT Do 29.10.
    result = calculate_deadline(
        "lieferbeginn",
        1,
        assignment_date=date(2026, 11, 2),
        step1_sent_at=datetime(2026, 10, 26, 17, 20),
    )
    assert result["latest_transmission_day"] == "2026-10-29"
    assert result["sent_in_time"] is True


def test_latest_deregistration_day_uses_zuordnungsende():
    result = calculate_deadline("lieferende_lf", 1, assignment_date=date(2026, 11, 1))
    assert result["latest_transmission_day"] == "2026-10-29"
    assert "Zuordnungsende" in result["explanation"]


def test_rejection_deadline_over_christmas():
    result = calculate_deadline("lieferbeginn", 6, step1_sent_at=datetime(2026, 12, 23, 10, 0))
    assert result["deadline"] == "2026-12-28T11:00"
    assert "Heiligabend" in result["explanation"]
    assert "Nr. 6" in result["source"]


def test_malo_id_answer_is_two_clock_hours_later():
    result = calculate_deadline("malo_ermittlung", 2, step1_sent_at=datetime(2026, 10, 30, 23, 30))
    assert result["deadline"] == "2026-10-31T01:30"


def test_termination_answer_is_due_on_next_working_day():
    result = calculate_deadline("kuendigung", 2, step1_sent_at=datetime(2026, 10, 30, 9, 0))
    assert result["deadline"] == "2026-11-02"


def test_default_supply_window_depends_on_whether_start_is_in_future():
    sent = datetime(2026, 10, 14, 11, 0)
    future = calculate_deadline("ersatzversorgung_beginn", 3, sent, date(2026, 10, 15))
    assert (future["earliest"], future["deadline"]) == ("2026-10-14T15:00", "2026-10-14T16:00")
    today = calculate_deadline("ersatzversorgung_beginn", 3, sent, date(2026, 10, 14))
    assert today["deadline"] == "2026-10-15T16:00"


def test_every_rule_cites_gpke_part_2_with_page():
    for rule in RULES:
        assert rule.source.startswith("GPKE Teil 2") and ", S. " in rule.source


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"process": "wechsel", "step": 1}, "Unbekannter Prozess"),
        ({"process": "lieferbeginn", "step": 3}, "Verfügbare Schritte: 1, 2, 4, 5, 6, 10, 13"),
        ({"process": "lieferbeginn", "step": 6}, "step1_sent_at"),
        ({"process": "lieferbeginn", "step": 1}, "assignment_date"),
    ],
)
def test_helpful_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        calculate_deadline(**kwargs)
