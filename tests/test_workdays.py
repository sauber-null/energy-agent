from datetime import date

import pytest

from energy_agent.tools.workdays import (
    easter_sunday,
    is_werktag,
    next_werktag,
    non_working_reason,
    previous_werktag,
    skipped_days,
)


def test_easter_sunday():
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)


@pytest.mark.parametrize(
    "day",
    [
        date(2025, 6, 9),  # Pfingstmontag
        date(2025, 6, 19),  # Fronleichnam: only some states, but counts nationwide
        date(2025, 11, 19),  # Buß- und Bettag (Saxony only)
        date(2025, 5, 8),  # one-off holiday in Berlin
        date(2027, 3, 8),  # Internationaler Frauentag (BE, MV), a Monday
        date(2026, 12, 24),  # GPKE: 24.12. counts as a holiday
        date(2026, 12, 31),  # GPKE: 31.12. counts as a holiday
        date(2026, 10, 31),  # Saturday
    ],
)
def test_non_working_days(day):
    assert not is_werktag(day)


def test_regular_and_former_one_off_days_are_working_days():
    assert is_werktag(date(2026, 10, 27))
    assert is_werktag(date(2026, 5, 8))  # Berlin's 8 May holiday was 2025 only
    assert non_working_reason(date(2026, 10, 27)) is None


def test_next_and_previous_werktag_skip_christmas_and_weekends():
    assert next_werktag(date(2026, 12, 23)) == date(2026, 12, 28)
    assert previous_werktag(date(2026, 11, 2)) == date(2026, 10, 30)
    assert next_werktag(date(2025, 6, 6)) == date(2025, 6, 10)


def test_skipped_days_names_the_reason():
    assert skipped_days(date(2025, 6, 6), date(2025, 6, 10)) == [
        "Sa 07.06.2025 (Samstag)",
        "So 08.06.2025 (Sonntag)",
        "Mo 09.06.2025 (Pfingstmontag)",
    ]


def test_years_before_2023_are_rejected():
    with pytest.raises(ValueError, match="2023"):
        is_werktag(date(2022, 6, 1))
