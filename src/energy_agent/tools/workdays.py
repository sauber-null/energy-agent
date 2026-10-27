"""Working days (Werktage, WT) as defined for German energy market communication.

GPKE Teil 1, Kap. 7 Fristenberechnung: a Werktag is any day that is not a Saturday, Sunday
or public holiday. A holiday in any single federal state counts as a holiday nationwide,
and 24.12. and 31.12. count as holidays. The calendar below is therefore the union of all
state holidays. It covers 2023 onwards (Frauentag in MV since 2023).
"""

from datetime import date, timedelta
from functools import cache

FIRST_SUPPORTED_YEAR = 2023

WEEKDAYS_DE = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")

# One-off holidays declared by a federal state.
ONE_OFF_HOLIDAYS = {
    date(2025, 5, 8): "80. Jahrestag des Kriegsendes (BE, einmalig)",
}


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    weekday_shift = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * weekday_shift) // 451
    month, day = divmod(h + weekday_shift - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _buss_und_bettag(year: int) -> date:
    """The Wednesday before 23 November."""
    nov22 = date(year, 11, 22)
    return nov22 - timedelta(days=(nov22.weekday() - 2) % 7)


@cache
def holidays(year: int) -> dict[date, str]:
    if year < FIRST_SUPPORTED_YEAR:
        raise ValueError(f"holiday calendar covers {FIRST_SUPPORTED_YEAR} onwards, not {year}")
    easter = easter_sunday(year)
    days = {
        date(year, 1, 1): "Neujahr",
        date(year, 1, 6): "Heilige Drei Könige (BW, BY, ST)",
        date(year, 3, 8): "Internationaler Frauentag (BE, MV)",
        easter - timedelta(days=2): "Karfreitag",
        easter + timedelta(days=1): "Ostermontag",
        date(year, 5, 1): "Tag der Arbeit",
        easter + timedelta(days=39): "Christi Himmelfahrt",
        easter + timedelta(days=50): "Pfingstmontag",
        easter + timedelta(days=60): "Fronleichnam (BW, BY, HE, NW, RP, SL)",
        date(year, 8, 15): "Mariä Himmelfahrt (SL)",
        date(year, 9, 20): "Weltkindertag (TH)",
        date(year, 10, 3): "Tag der Deutschen Einheit",
        date(year, 10, 31): "Reformationstag (BB, HB, HH, MV, NI, SN, ST, SH, TH)",
        date(year, 11, 1): "Allerheiligen (BW, BY, NW, RP, SL)",
        _buss_und_bettag(year): "Buß- und Bettag (SN)",
        date(year, 12, 24): "Heiligabend (gilt laut GPKE als Feiertag)",
        date(year, 12, 25): "1. Weihnachtstag",
        date(year, 12, 26): "2. Weihnachtstag",
        date(year, 12, 31): "Silvester (gilt laut GPKE als Feiertag)",
    }
    days.update({day: name for day, name in ONE_OFF_HOLIDAYS.items() if day.year == year})
    return days


def non_working_reason(day: date) -> str | None:
    """Why `day` is not a Werktag, or None if it is one."""
    if day.weekday() == 5:
        return "Samstag"
    if day.weekday() == 6:
        return "Sonntag"
    return holidays(day.year).get(day)


def is_werktag(day: date) -> bool:
    return non_working_reason(day) is None


def next_werktag(day: date) -> date:
    """The first Werktag strictly after `day` ("1. WT nach ...")."""
    day += timedelta(days=1)
    while not is_werktag(day):
        day += timedelta(days=1)
    return day


def previous_werktag(day: date) -> date:
    """The last Werktag strictly before `day` ("letzter WT vor ...")."""
    day -= timedelta(days=1)
    while not is_werktag(day):
        day -= timedelta(days=1)
    return day


def skipped_days(after: date, before: date) -> list[str]:
    """Non-working days strictly between two dates, e.g. ['Sa 07.06.2025 (Samstag)']."""
    skipped = []
    day = after + timedelta(days=1)
    while day < before:
        if reason := non_working_reason(day):
            skipped.append(f"{format_date(day)} ({reason})")
        day += timedelta(days=1)
    return skipped


def format_date(day: date) -> str:
    return f"{WEEKDAYS_DE[day.weekday()]} {day:%d.%m.%Y}"
