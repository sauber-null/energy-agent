import re
import tomllib
from pathlib import Path

SOURCES = Path(__file__).resolve().parent.parent / "data" / "sources.toml"


def load_documents() -> list[dict]:
    return tomllib.loads(SOURCES.read_text())["documents"]


def test_ids_and_files_are_unique():
    documents = load_documents()
    assert len({d["id"] for d in documents}) == len(documents)
    assert len({d["file"] for d in documents}) == len(documents)


def test_sources_are_official_bnetza_pdfs():
    for doc in load_documents():
        assert doc["url"].startswith("https://www.bundesnetzagentur.de/")
        assert doc["file"].endswith(".pdf")


def test_every_document_has_a_pinned_checksum():
    for doc in load_documents():
        assert re.fullmatch(r"[0-9a-f]{64}", doc["sha256"]), doc["id"]
