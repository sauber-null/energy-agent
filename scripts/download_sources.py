"""Download the source PDFs listed in data/sources.toml into data/raw/.

Usage: uv run python scripts/download_sources.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

from energy_agent.sources import RAW_DIR, load_sources

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
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    mismatches = 0
    for source in load_sources():
        if not source.path.exists():
            print(f"downloading {source.id} ...")
            download(source.url, source.path)
        actual = sha256(source.path)
        if source.sha256 and actual != source.sha256:
            print(f"CHECKSUM MISMATCH {source.id}: expected {source.sha256}, got {actual}")
            mismatches += 1
        else:
            print(f"ok {source.id}  {source.path.stat().st_size // 1024} KB  sha256={actual}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
