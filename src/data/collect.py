"""Download raw Champions League match data and save it unchanged to data/raw/.

Phase: Milestone 2 - Data collection.

Previous stage: none. This is the entry point of the pipeline.
Next stage: clean.py parses the files written here into a match table.

Historical results come from openfootball/champions-league (CC0-1.0 public domain),
one Football.TXT file per season. Files are saved byte-for-byte as downloaded and
never edited. Every download is recorded in a log (URL, UTC timestamp, size, SHA-256)
so each file can be traced back to its source and checked for later changes.

Run from the project root:  python -m src.data.collect

TODO (Part C, current-season fixtures from football-data.org):
- load_api_key(var_name: str) -> str
    Read a key from the environment (.env); raise a clear error if it is missing.
- fetch_fixtures(api_key: str) -> dict
    Request the current season's matches; saved to a git-ignored folder.
"""

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import HISTORICAL_SEASONS, OPENFOOTBALL_BASE_URL, OPENFOOTBALL_RAW_DIR

SEASON_PATTERN = re.compile(r"^\d{4}-\d{2}$")  # e.g. "2025-26"
LOG_FILENAME = "download_log.csv"
LOG_COLUMNS = ["season", "url", "file", "bytes", "sha256", "downloaded_at_utc"]


def build_season_url(season: str) -> str:
    """Return the raw-file URL of one season's Champions League file.

    Args:
        season: Season in "YYYY-YY" form, e.g. "2025-26".

    Raises:
        ValueError: If the season is not in "YYYY-YY" form or the years don't follow on.
    """
    if not SEASON_PATTERN.match(season):
        raise ValueError(f"Season must look like '2025-26', got {season!r}")
    start, end = season.split("-")
    if (int(start) + 1) % 100 != int(end):
        raise ValueError(f"Season years must be consecutive, got {season!r}")
    return f"{OPENFOOTBALL_BASE_URL}/{season}/cl.txt"


def download_bytes(url: str, timeout: int = 30) -> bytes:
    """Download a URL and return its raw bytes; raise on HTTP errors (e.g. 404)."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def append_log_row(log_path: Path, row: dict[str, str | int]) -> None:
    """Append one download record to the CSV log, writing the header if the file is new."""
    is_new = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def collect_season(season: str, out_dir: Path, overwrite: bool = False) -> Path:
    """Download one season's file into out_dir unchanged and log the download.

    Args:
        season: Season in "YYYY-YY" form.
        out_dir: Folder for raw files (created if missing).
        overwrite: Re-download even if the file already exists.

    Returns:
        Path of the saved file, e.g. out_dir / "2025-26_cl.txt".
    """
    url = build_season_url(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{season}_cl.txt"
    if path.exists() and not overwrite:
        return path

    content = download_bytes(url)
    path.write_bytes(content)  # bytes, not text: no encoding or line-ending changes
    append_log_row(
        out_dir / LOG_FILENAME,
        {
            "season": season,
            "url": url,
            "file": path.name,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return path


def collect_seasons(
    seasons: list[str] = HISTORICAL_SEASONS,
    out_dir: Path = OPENFOOTBALL_RAW_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Collect several seasons; return the saved paths in season order."""
    return [collect_season(season, out_dir, overwrite) for season in seasons]


if __name__ == "__main__":
    for saved in collect_seasons():
        print(f"{saved.name}: {saved.stat().st_size:,} bytes")
