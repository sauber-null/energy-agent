"""Download the source PDFs listed in data/sources.toml into data/raw/.

Usage: uv run python scripts/download_sources.py
"""

import hashlib
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources.toml"
RAW_DIR = ROOT / "data" / "raw"

# bundesnetzagentur.de rejects some non-browser clients, so send a browser User-Agent.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.0 Safari/605.1.15"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"{url} did not return a PDF")
    target.write_bytes(data)


def main() -> int:
    documents = tomllib.loads(SOURCES.read_text())["documents"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    mismatches = 0
    for doc in documents:
        target = RAW_DIR / doc["file"]
        if not target.exists():
            print(f"downloading {doc['id']} ...")
            download(doc["url"], target)
        actual = sha256(target)
        expected = doc.get("sha256")
        if expected and actual != expected:
            print(f"CHECKSUM MISMATCH {doc['id']}: expected {expected}, got {actual}")
            mismatches += 1
        else:
            print(f"ok {doc['id']}  {target.stat().st_size // 1024} KB  sha256={actual}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
