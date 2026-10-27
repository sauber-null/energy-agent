"""The document corpus: metadata from data/sources.toml and local file locations."""

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SOURCES_FILE = DATA_DIR / "sources.toml"


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    file: str
    url: str
    valid_from: date
    sha256: str

    @property
    def path(self) -> Path:
        return RAW_DIR / self.file


def load_sources(path: Path = SOURCES_FILE) -> list[Source]:
    return [Source(**doc) for doc in tomllib.loads(path.read_text())["documents"]]
