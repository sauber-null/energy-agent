"""A small test database of supplier switch requests (fictional companies and IDs)."""

from datetime import date, datetime

import psycopg

SCHEMA = """
CREATE TABLE IF NOT EXISTS switch_requests (
    request_id       text PRIMARY KEY,
    malo_id          char(11)  NOT NULL,
    process          text      NOT NULL,  -- lieferbeginn | lieferende_lf | ersatzversorgung_beginn
    status           text      NOT NULL,  -- angemeldet | bestaetigt | abgelehnt
    supplier_new     text,
    supplier_old     text,
    grid_operator    text      NOT NULL,
    customer_type    text      NOT NULL,  -- haushaltskunde | gewerbe
    assignment_date  date      NOT NULL,  -- Zuordnungsbeginn, or Zuordnungsende for lieferende
    submitted_at     timestamp NOT NULL,  -- ÜZ of step 1
    updated_at       timestamp NOT NULL,
    note             text                 -- rejection reason or remark
);
"""

STATUS_TEXT = {
    "angemeldet": "Nachricht aus Schritt 1 an den NB übermittelt, Antwort des NB steht aus.",
    "bestaetigt": "Vom NB bestätigt, die Zuordnung ist wirksam zum angegebenen Datum.",
    "abgelehnt": "Vom NB abgelehnt, siehe note für den Grund.",
}

_GRID = "Testnetz Rhein GmbH"
# fmt: off
SEED = [
    ("LW-2026-0001", "51234567895", "lieferbeginn", "bestaetigt", "Muster Energie GmbH",
     "Beispiel Stadtwerke AG", _GRID, "haushaltskunde", date(2026, 10, 21),
     datetime(2026, 10, 19, 9, 12), datetime(2026, 10, 20, 8, 30), None),
    ("LW-2026-0002", "59876543215", "lieferbeginn", "abgelehnt", "Muster Energie GmbH",
     None, _GRID, "haushaltskunde", date(2026, 10, 27),
     datetime(2026, 10, 22, 14, 5), datetime(2026, 10, 23, 10, 41),
     "Marktlokation nicht eindeutig identifizierbar"),
    ("LW-2026-0003", "51002003005", "lieferbeginn", "angemeldet", "Nordwind Strom GmbH",
     "Muster Energie GmbH", _GRID, "gewerbe", date(2026, 11, 2),
     datetime(2026, 10, 26, 17, 20), datetime(2026, 10, 26, 17, 20), None),
    ("LE-2026-0004", "55555123451", "lieferende_lf", "bestaetigt", None,
     "Nordwind Strom GmbH", _GRID, "gewerbe", date(2026, 11, 1),
     datetime(2026, 10, 20, 10, 0), datetime(2026, 10, 21, 5, 48), None),
    ("EV-2026-0005", "52468135797", "ersatzversorgung_beginn", "bestaetigt",
     "Beispiel Stadtwerke AG", "Sonnental Energie AG", _GRID, "haushaltskunde",
     date(2026, 10, 15), datetime(2026, 10, 14, 11, 0), datetime(2026, 10, 14, 14, 20),
     "Lieferende ohne Nachfolgelieferant, Kunde in Ersatzversorgung"),
    ("LW-2026-0006", "53101010105", "lieferbeginn", "abgelehnt", "Nordwind Strom GmbH",
     "Sonnental Energie AG", _GRID, "haushaltskunde", date(2026, 10, 23),
     datetime(2026, 10, 21, 16, 30), datetime(2026, 10, 22, 10, 55),
     "Widerspruch des LFA: Vertrag mit Mindestlaufzeit bis 31.12.2026"),
    ("LW-2026-0007", "57720033118", "lieferbeginn", "angemeldet", "Muster Energie GmbH",
     "Nordwind Strom GmbH", _GRID, "haushaltskunde", date(2026, 11, 1),
     datetime(2026, 10, 27, 8, 15), datetime(2026, 10, 27, 8, 15), None),
]
# fmt: on

_COLUMNS = (
    "request_id",
    "malo_id",
    "process",
    "status",
    "supplier_new",
    "supplier_old",
    "grid_operator",
    "customer_type",
    "assignment_date",
    "submitted_at",
    "updated_at",
    "note",
)


def is_valid_malo_id(malo_id: str) -> bool:
    """11 digits, the last one a check digit (BDEW): odd positions + 2 x even positions."""
    if len(malo_id) != 11 or not malo_id.isdigit():
        return False
    digits = [int(d) for d in malo_id[:10]]
    total = sum(digits[0::2]) + 2 * sum(digits[1::2])
    return (10 - total % 10) % 10 == int(malo_id[10])


def seed(conn: psycopg.Connection) -> int:
    with conn.transaction():
        conn.execute(SCHEMA)
        conn.execute("DELETE FROM switch_requests")
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(_COLUMNS))
            cur.executemany(
                f"INSERT INTO switch_requests ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                SEED,
            )
    return len(SEED)


def get_switch_status(
    conn: psycopg.Connection, request_id: str | None = None, malo_id: str | None = None
) -> dict:
    if malo_id is not None and not is_valid_malo_id(malo_id):
        raise ValueError(f"'{malo_id}' ist keine gültige MaLo-ID (11 Ziffern mit Prüfziffer).")
    if request_id:
        rows = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM switch_requests WHERE request_id = %s",
            (request_id.strip().upper(),),
        ).fetchall()
    elif malo_id:
        rows = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM switch_requests WHERE malo_id = %s "
            "ORDER BY submitted_at DESC",
            (malo_id,),
        ).fetchall()
    else:
        raise ValueError("request_id oder malo_id wird benötigt.")

    requests = []
    for row in rows:
        record = {
            col: value.isoformat() if isinstance(value, date | datetime) else value
            for col, value in zip(_COLUMNS, row, strict=True)
        }
        record["status_text"] = STATUS_TEXT[record["status"]]
        requests.append(record)
    return {"found": len(requests), "requests": requests}
