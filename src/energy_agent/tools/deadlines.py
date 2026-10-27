"""Deadlines (Fristen) of GPKE process steps (LFW24 version, in force since 2025-06-06).

Each rule restates one "Frist" cell of the process tables in GPKE Teil 2 and cites it, so the
agent can quote its source. Only the rules for non-EEG market locations are covered.
Times are German local time. Abbreviations as in the GPKE: ÜZ = Übermittlungszeitpunkt
(time a message is sent), ÜT = Übermittlungstag (day it is sent), WT = Werktag.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import Literal

from energy_agent.tools.workdays import format_date, next_werktag, previous_werktag, skipped_days

Process = Literal[
    "malo_ermittlung",
    "kuendigung",
    "lieferbeginn",
    "lieferende_lf",
    "lieferende_nb",
    "ersatzversorgung_beginn",
]


class RuleKind(StrEnum):
    # Step 1 An-/Abmeldung: "spätester ÜT ist der Tag vor dem letzten WT vor dem
    # Zuordnungsbeginn" (or Zuordnungsende).
    BEFORE_ASSIGNMENT = "before_assignment"
    # "spätester ÜZ ist HH:MM Uhr des 1. WT nach dem ÜT von Nr. 1"
    NEXT_WT_AT = "next_wt_at"
    # "spätester ÜT ist der 1. WT nach dem ÜT von Nr. 1"
    NEXT_WT_DAY = "next_wt_day"
    # "spätester ÜZ ist N Stunden nach dem ÜZ von Nr. 1"
    HOURS_AFTER = "hours_after"
    # Ersatz-/Grundversorgung: HH:MM am ÜT von Nr. 1 if the Zuordnungsbeginn lies in the
    # future, otherwise HH:MM des 1. WT nach dem ÜT von Nr. 1.
    SAME_DAY_OR_NEXT_WT_AT = "same_day_or_next_wt_at"


@dataclass(frozen=True)
class DeadlineRule:
    process: Process
    step: int
    action: str
    kind: RuleKind
    rule_text: str
    source: str
    latest: time | None = None
    earliest: time | None = None
    hours: int = 0


_TEIL2 = "GPKE Teil 2 (Anlage 1b zu BK6-22-024)"

# fmt: off
RULES = [
    DeadlineRule(
        "malo_ermittlung", 2, "Rückmeldung auf Anfrage der MaLo-ID (NB an LF)",
        RuleKind.HOURS_AFTER, "Unverzüglich, jedoch spätester ÜZ ist 2 Stunden nach dem ÜZ von Nr. 1.",
        f"{_TEIL2}, 1.1.2 SD: Ermittlung der MaLo-ID, Nr. 2, S. 6", hours=2,
    ),
    DeadlineRule(
        "kuendigung", 2, "Antwort des LFA auf die Kündigung",
        RuleKind.NEXT_WT_DAY, "Unverzüglich, jedoch spätester ÜT ist der 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 1.2.2 SD: Kündigung, Nr. 2, S. 9",
    ),
    DeadlineRule(
        "lieferbeginn", 1, "Anmeldung einer Zuordnung des LFN zur Marktlokation (LFN an NB)",
        RuleKind.BEFORE_ASSIGNMENT,
        "Unverzüglich nach Vorliegen des Anmeldegrundes, jedoch spätester ÜT ist der Tag vor dem "
        "letzten WT vor dem Zuordnungsbeginn (nicht EEG-Marktlokationen).",
        f"{_TEIL2}, 2.1.2 SD: Lieferbeginn, Nr. 1, S. 17",
    ),
    DeadlineRule(
        "lieferbeginn", 2, "Information über existierende Zuordnung (NB an LFA)",
        RuleKind.NEXT_WT_AT, "Unverzüglich, jedoch spätester ÜZ ist 07:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.1.2 SD: Lieferbeginn, Nr. 2, S. 19", latest=time(7),
    ),
    DeadlineRule(
        "lieferbeginn", 4, "Antwort des LFA auf Anfrage zur Beendigung seiner Zuordnung",
        RuleKind.NEXT_WT_AT, "Unverzüglich, jedoch spätester ÜZ ist 09:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.1.2 SD: Lieferbeginn, Nr. 4, S. 20", latest=time(9),
    ),
    DeadlineRule(
        "lieferbeginn", 5, "Zuordnung des LFN zur Marktlokation (Bestätigung durch NB)",
        RuleKind.NEXT_WT_AT, "Unverzüglich, jedoch spätester ÜZ ist 11:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.1.2 SD: Lieferbeginn, Nr. 5, S. 20-21", latest=time(11),
    ),
    DeadlineRule(
        "lieferbeginn", 6, "Ablehnung der Anmeldung durch den NB",
        RuleKind.NEXT_WT_AT, "Unverzüglich, jedoch spätester ÜZ ist 11:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.1.2 SD: Lieferbeginn, Nr. 6, S. 21", latest=time(11),
    ),
    DeadlineRule(
        "lieferbeginn", 10, "Beendigung der Zuordnung des LFA (NB an LFA)",
        RuleKind.NEXT_WT_AT, "Unverzüglich nach dem ÜZ von Nr. 5, jedoch spätester ÜZ ist 12:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.1.2 SD: Lieferbeginn, Nr. 10, S. 22", latest=time(12),
    ),
    DeadlineRule(
        "lieferbeginn", 13, "Aufhebung der Zuordnung des LFZ (NB an LFZ)",
        RuleKind.NEXT_WT_AT, "Unverzüglich nach dem ÜZ von Nr. 5, jedoch spätester ÜZ ist 12:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.1.2 SD: Lieferbeginn, Nr. 13, S. 23", latest=time(12),
    ),
    DeadlineRule(
        "lieferende_lf", 1, "Abmeldung einer Zuordnung des LF (LF an NB)",
        RuleKind.BEFORE_ASSIGNMENT,
        "Unverzüglich nach Vorliegen des Abmeldegrundes, jedoch spätester ÜT ist der Tag vor dem "
        "letzten WT vor dem Zuordnungsende (nicht EEG-Marktlokationen).",
        f"{_TEIL2}, 2.5.1.2 SD: Lieferende von LF an NB, Nr. 1, S. 53",
    ),
    DeadlineRule(
        "lieferende_lf", 2, "Beendigung der Zuordnung des LF (Bestätigung durch NB)",
        RuleKind.NEXT_WT_AT, "Unverzüglich, jedoch spätester ÜZ ist 06:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.5.1.2 SD: Lieferende von LF an NB, Nr. 2, S. 53", latest=time(6),
    ),
    DeadlineRule(
        "lieferende_lf", 3, "Ablehnung der Abmeldung durch den NB",
        RuleKind.NEXT_WT_AT, "Unverzüglich, jedoch spätester ÜZ ist 06:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.5.1.2 SD: Lieferende von LF an NB, Nr. 3, S. 53", latest=time(6),
    ),
    DeadlineRule(
        "lieferende_nb", 2, "Antwort des LF auf die Ankündigung der Beendigung durch den NB",
        RuleKind.NEXT_WT_AT, "Unverzüglich, jedoch spätester ÜZ ist 05:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.5.2.2 SD: Lieferende von NB an LF, Nr. 2, S. 58", latest=time(5),
    ),
    DeadlineRule(
        "lieferende_nb", 3, "Beendigung der Zuordnung des LF durch den NB",
        RuleKind.NEXT_WT_AT,
        "Unverzüglich, jedoch frühester ÜZ ist 05:00 Uhr und spätester 06:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.5.2.2 SD: Lieferende von NB an LF, Nr. 3, S. 58", latest=time(6), earliest=time(5),
    ),
    DeadlineRule(
        "ersatzversorgung_beginn", 2, "Antwort des E/G auf die Ankündigung seiner Zuordnung",
        RuleKind.SAME_DAY_OR_NEXT_WT_AT,
        "Zuordnungsbeginn in der Zukunft: spätester ÜZ ist 15:00 Uhr am ÜT von Nr. 1; sonst "
        "spätester ÜZ ist 15:00 Uhr des 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.3.2.2 SD: Beginn der Ersatz-/Grundversorgung, Nr. 2, S. 34", latest=time(15),
    ),
    DeadlineRule(
        "ersatzversorgung_beginn", 3, "Zuordnung des E/G durch den NB bei fehlender Antwort",
        RuleKind.SAME_DAY_OR_NEXT_WT_AT,
        "Zuordnungsbeginn in der Zukunft: frühester ÜZ 15:00 Uhr, spätester 16:00 Uhr am ÜT von "
        "Nr. 1; sonst dieselben Uhrzeiten am 1. WT nach dem ÜT von Nr. 1.",
        f"{_TEIL2}, 2.3.2.2 SD: Beginn der Ersatz-/Grundversorgung, Nr. 3, S. 34",
        latest=time(16), earliest=time(15),
    ),
]
# fmt: on


def find_rule(process: str, step: int) -> DeadlineRule:
    for rule in RULES:
        if rule.process == process and rule.step == step:
            return rule
    steps = sorted(r.step for r in RULES if r.process == process)
    if not steps:
        processes = sorted({r.process for r in RULES})
        raise ValueError(f"Unbekannter Prozess '{process}'. Verfügbar: {', '.join(processes)}")
    raise ValueError(
        f"Für '{process}' ist keine Frist zu Schritt {step} hinterlegt. "
        f"Verfügbare Schritte: {', '.join(map(str, steps))}"
    )


def _fmt(moment: datetime) -> str:
    return f"{format_date(moment.date())}, {moment:%H:%M} Uhr"


def _next_wt_explanation(sent_day: date, target: date) -> str:
    skipped = skipped_days(sent_day, target)
    gap = f" (übersprungen: {', '.join(skipped)})" if skipped else ""
    return f"ÜT von Nr. 1: {format_date(sent_day)}. 1. WT danach: {format_date(target)}{gap}."


def calculate_deadline(
    process: str,
    step: int,
    step1_sent_at: datetime | None = None,
    assignment_date: date | None = None,
) -> dict:
    """Deadline of one process step.

    step1_sent_at:   ÜZ of the step 1 message (Anmeldung, Abmeldung, Anfrage, Ankündigung ...).
    assignment_date: Zuordnungsbeginn (Lieferbeginn, Ersatzversorgung) or Zuordnungsende
                     (Lieferende). For step 1 of lieferbeginn/lieferende, step1_sent_at gives
                     the earliest possible date and assignment_date the latest ÜT; with both,
                     both are returned plus whether step 1 was sent in time.
    """
    rule = find_rule(process, step)
    result = {
        "process": rule.process,
        "step": rule.step,
        "action": rule.action,
        "rule": rule.rule_text,
        "source": rule.source,
    }

    if rule.kind is RuleKind.BEFORE_ASSIGNMENT:
        noun = "Zuordnungsende" if process == "lieferende_lf" else "Zuordnungsbeginn"
        if not (assignment_date or step1_sent_at):
            raise ValueError(
                f"Für Schritt 1 wird assignment_date ({noun}) oder step1_sent_at benötigt."
            )
        # Small models sometimes fill both fields, so compute whatever is given instead of
        # letting one silently win.
        explanations = []
        if step1_sent_at:
            first_wt = next_werktag(step1_sent_at.date())
            earliest = first_wt + timedelta(days=1)
            result[f"earliest_{'end' if noun == 'Zuordnungsende' else 'start'}"] = (
                earliest.isoformat()
            )
            explanations.append(
                f"{_next_wt_explanation(step1_sent_at.date(), first_wt)} Frühester {noun} ist der "
                f"Tag danach: {format_date(earliest)}, 00:00 Uhr."
            )
        if assignment_date:
            last_wt = previous_werktag(assignment_date)
            latest_day = last_wt - timedelta(days=1)
            result["latest_transmission_day"] = latest_day.isoformat()
            explanations.append(
                f"{noun}: {format_date(assignment_date)}. Letzter WT davor: {format_date(last_wt)}. "
                f"Der Tag davor ist der späteste ÜT: {format_date(latest_day)}."
            )
            if step1_sent_at:
                result["sent_in_time"] = step1_sent_at.date() <= latest_day
        result["explanation"] = " ".join(explanations)
        return result

    if step1_sent_at is None:
        raise ValueError("step1_sent_at (Übermittlungszeitpunkt von Schritt 1) wird benötigt.")
    sent_day = step1_sent_at.date()

    if rule.kind is RuleKind.HOURS_AFTER:
        deadline = step1_sent_at + timedelta(hours=rule.hours)
        result["deadline"] = deadline.isoformat(timespec="minutes")
        result["explanation"] = (
            f"ÜZ von Nr. 1: {_fmt(step1_sent_at)} + {rule.hours} Stunden = {_fmt(deadline)}."
        )
        return result

    if rule.kind is RuleKind.NEXT_WT_DAY:
        day = next_werktag(sent_day)
        result["deadline"] = day.isoformat()
        result["explanation"] = (
            f"{_next_wt_explanation(sent_day, day)} Spätester ÜT: {format_date(day)}."
        )
        return result

    if rule.kind is RuleKind.SAME_DAY_OR_NEXT_WT_AT:
        if assignment_date is None:
            raise ValueError("assignment_date (Zuordnungsbeginn des E/G) wird benötigt.")
        if assignment_date > sent_day:
            day = sent_day
            explanation = f"Zuordnungsbeginn {format_date(assignment_date)} liegt in der Zukunft: Frist am ÜT von Nr. 1."
        else:
            day = next_werktag(sent_day)
            explanation = (
                f"Zuordnungsbeginn {format_date(assignment_date)} liegt nicht in der Zukunft. "
                + _next_wt_explanation(sent_day, day)
            )
    else:  # NEXT_WT_AT
        day = next_werktag(sent_day)
        explanation = _next_wt_explanation(sent_day, day)

    deadline = datetime.combine(day, rule.latest)
    result["deadline"] = deadline.isoformat(timespec="minutes")
    if rule.earliest:
        result["earliest"] = datetime.combine(day, rule.earliest).isoformat(timespec="minutes")
        explanation += f" Zeitfenster: {rule.earliest:%H:%M} bis {rule.latest:%H:%M} Uhr."
    result["explanation"] = f"{explanation} Spätester ÜZ: {_fmt(deadline)}."
    return result
