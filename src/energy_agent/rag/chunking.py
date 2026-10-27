"""Split Markdown pages into section-aware chunks.

Every heading closes the current chunk and updates the section path stored with each
chunk, e.g. "2 Zuordnungsprozesse > 2.1 Use-Case: Lieferbeginn > 2.1.1 UC: Lieferbeginn".
Within a section, paragraphs and table rows are packed into chunks of about `max_chars`.
A chunk that contains table rows starts with the table's header row, so a row such as
"1 | Anmeldung ... | spätestens ..." keeps its column names (Nr. | Aktion | Frist | ...).
"""

import re
from dataclasses import dataclass

_MARKUP = re.compile(r"\*\*|</?u>|`|<!-- (Start|End) of picture text -->")
_FOOTERS = (
    re.compile(r"^Seite \d+ von \d+$"),  # GPKE annexes
    re.compile(r"^- \d+ -$"),  # BNetzA decision
)
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")
_MAX_HEADER_CHARS = 200
_SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")

# Heading kinds, outermost first. A heading closes every open heading of the same or a
# deeper kind: "A." > "I." > "1" > "1.1" > ... > unnumbered ("Hinweise:").
_LETTER = re.compile(r"^[A-Z]\.(\s|$)")
_ROMAN = re.compile(r"^[IVX]+\.(\s|$)")
_ARABIC = re.compile(r"^(\d+(?:\.\d+)*)\.?(\s|$)")
_UNNUMBERED_RANK = 100


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    section: str
    page_start: int
    page_end: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc_id}:{self.chunk_index:04d}"


@dataclass(frozen=True)
class _Unit:
    """A paragraph or table row: the smallest piece of text that is never split if it fits."""

    text: str
    page: int
    table_header: str | None = None


def clean_line(line: str) -> str:
    line = _MARKUP.sub("", line).replace("<br>", " ")
    return " ".join(line.split())


def is_footer(line: str) -> bool:
    return any(pattern.match(line) for pattern in _FOOTERS)


def heading_rank(title: str) -> int:
    if _ROMAN.match(title):  # checked before letters: "I." is a roman numeral here
        return 1
    if _LETTER.match(title):
        return 0
    if match := _ARABIC.match(title):
        return 2 + match.group(1).count(".")
    return _UNNUMBERED_RANK


def _table_row(line: str) -> str:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return " | ".join(cell for cell in cells if cell)


def _split_oversized(unit: _Unit, max_chars: int) -> list[_Unit]:
    if len(unit.text) <= max_chars:
        return [unit]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(unit.text):
        # Sentences longer than max_chars are cut at word boundaries.
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return [_Unit(piece, unit.page, unit.table_header) for piece in pieces]


class _ChunkBuilder:
    def __init__(self, doc_id: str, max_chars: int):
        self.doc_id = doc_id
        self.max_chars = max_chars
        self.chunks: list[Chunk] = []
        self.section = ""
        self._units: list[_Unit] = []
        self._carried = 0  # units repeated from the previous chunk as overlap

    def add(self, unit: _Unit) -> None:
        for piece in _split_oversized(unit, self.max_chars):
            if self._size() + len(piece.text) > self.max_chars:
                if len(self._units) > self._carried:
                    self.flush(overlap=True)
                if self._size() + len(piece.text) > self.max_chars:
                    self._units, self._carried = [], 0  # no room for the overlap
            self._units.append(piece)

    def flush(self, overlap: bool = False) -> None:
        if len(self._units) > self._carried:
            self.chunks.append(
                Chunk(
                    doc_id=self.doc_id,
                    chunk_index=len(self.chunks),
                    section=self.section,
                    page_start=self._units[0].page,
                    page_end=self._units[-1].page,
                    text=self._text(),
                )
            )
        last = self._units[-1] if self._units else None
        # Repeat a short last unit at the start of the next chunk so that a sentence
        # referring to the previous one keeps some context.
        if overlap and last and len(last.text) <= self.max_chars // 3:
            self._units, self._carried = [last], 1
        else:
            self._units, self._carried = [], 0

    def _size(self) -> int:
        return sum(len(u.text) + 1 for u in self._units)

    def _text(self) -> str:
        lines = [u.text for u in self._units]
        header = next((u.table_header for u in self._units if u.table_header), None)
        if header and header not in lines:
            lines.insert(0, header)
        return "\n".join(lines)


def chunk_document(doc_id: str, pages: list[str], max_chars: int = 1500) -> list[Chunk]:
    builder = _ChunkBuilder(doc_id, max_chars)
    headings: list[tuple[int, str]] = []
    table_header: str | None = None

    for page_no, page in enumerate(pages, start=1):
        lines = [clean_line(line) for line in page.splitlines()]
        paragraph: list[str] = []

        for i, line in enumerate(lines):
            if not line or is_footer(line):
                _end_paragraph(builder, paragraph, page_no)
            elif line.startswith("#"):
                _end_paragraph(builder, paragraph, page_no)
                builder.flush()
                title = line.lstrip("#").strip()
                rank = heading_rank(title)
                while headings and headings[-1][0] >= rank:
                    headings.pop()
                headings.append((rank, title))
                builder.section = " > ".join(t for _, t in headings)
                table_header = None
            elif _TABLE_SEPARATOR.match(line):
                continue
            elif line.startswith("|"):
                _end_paragraph(builder, paragraph, page_no)
                row = _table_row(line)
                next_line = next((n for n in lines[i + 1 :] if n), "")
                # A long "header" is a data row the parser mistook for one: keep it as content.
                if _TABLE_SEPARATOR.match(next_line) and len(row) <= _MAX_HEADER_CHARS:
                    table_header = row
                elif row:
                    builder.add(_Unit(row, page_no, table_header))
            else:
                if not paragraph:
                    table_header = None
                paragraph.append(line)
        _end_paragraph(builder, paragraph, page_no)

    builder.flush()
    return builder.chunks


def _end_paragraph(builder: _ChunkBuilder, paragraph: list[str], page_no: int) -> None:
    if paragraph:
        builder.add(_Unit(" ".join(paragraph), page_no))
        paragraph.clear()
