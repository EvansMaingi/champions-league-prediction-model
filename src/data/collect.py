"""Download raw Champions League match data and save it unchanged to data/raw/.

Phase: Milestone 2 - Data collection.

Previous stage: none. This is the entry point of the pipeline.
Next stage: clean.py parses the files written here into a match table.

Sources:
- Historical results: openfootball/champions-league (CC0-1.0 public domain), one
  Football.TXT file per season, committed to git.
- Current-season fixtures: football-data.org API (free plan). Each fetch is saved as a
  new timestamped snapshot, because fixtures change; the folder is git-ignored because
  the provider publishes no redistribution terms.
- Quality-check pages: English Wikipedia season articles (CC BY-SA 4.0), used to
  cross-check scores, kickoff times, venues and attendances; git-ignored.

Files are saved byte-for-byte as downloaded and never edited. Every download is
recorded in a log (URL, UTC timestamp, size, SHA-256) so each file can be traced back
to its source and checked for later changes. API keys never appear in logs or output.

Run from the project root:
    python -m src.data.collect             # historical seasons (default)
    python -m src.data.collect fixtures    # one current-season fixture snapshot
    python -m src.data.collect wikipedia   # Wikipedia pages for quality checks
"""

import argparse
import csv
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.config import (
    ENV_FILE,
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_KEY_VAR,
    FOOTBALL_DATA_RAW_DIR,
    HISTORICAL_SEASONS,
    HTTP_USER_AGENT,
    OPENFOOTBALL_BASE_URL,
    OPENFOOTBALL_RAW_DIR,
    WIKIPEDIA_API_URL,
    WIKIPEDIA_RAW_DIR,
)

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


def download_bytes(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> bytes:
    """Download a URL and return its raw bytes; raise on HTTP errors (e.g. 403, 404)."""
    response = requests.get(url, headers=headers, timeout=timeout)
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


def save_and_log(content: bytes, path: Path, season: str, url: str, now: datetime) -> Path:
    """Write content to path unchanged and append its record to the folder's log."""
    path.write_bytes(content)  # bytes, not text: no encoding or line-ending changes
    append_log_row(
        path.parent / LOG_FILENAME,
        {
            "season": season,
            "url": url,
            "file": path.name,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "downloaded_at_utc": now.isoformat(timespec="seconds"),
        },
    )
    return path


# --- Historical results (openfootball) --------------------------------------


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
    return save_and_log(content, path, season, url, datetime.now(timezone.utc))


def collect_seasons(
    seasons: list[str] = HISTORICAL_SEASONS,
    out_dir: Path = OPENFOOTBALL_RAW_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Collect several seasons; return the saved paths in season order."""
    return [collect_season(season, out_dir, overwrite) for season in seasons]


# --- Current-season fixtures (football-data.org) -----------------------------


def load_api_key(var_name: str = FOOTBALL_DATA_KEY_VAR, env_file: Path = ENV_FILE) -> str:
    """Return an API key from the environment, loading .env first if it exists.

    Raises:
        RuntimeError: If the variable is missing or empty, with instructions to fix it.
    """
    load_dotenv(env_file)
    key = os.getenv(var_name, "").strip()
    if not key:
        raise RuntimeError(
            f"{var_name} is not set. Add the line '{var_name}=<your token>' to {env_file} "
            "(see .env.example)."
        )
    return key


def fixtures_url(competition: str = "CL") -> str:
    """Return the football-data.org URL for a competition's current-season matches."""
    return f"{FOOTBALL_DATA_BASE_URL}/competitions/{competition}/matches"


def collect_fixtures(
    api_key: str,
    out_dir: Path = FOOTBALL_DATA_RAW_DIR,
    now: datetime | None = None,
) -> Path:
    """Fetch current-season matches and save them as a new timestamped JSON snapshot.

    Args:
        api_key: football-data.org token; sent only in the X-Auth-Token header.
        out_dir: Folder for snapshots (created if missing; git-ignored).
        now: UTC fetch time (defaults to the real clock; tests pass a fixed time).

    Returns:
        Path of the snapshot, e.g. out_dir / "cl_matches_20261020T170600Z.json".

    Raises:
        FileExistsError: If a snapshot with the same timestamp exists; never overwritten.
    """
    now = now or datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cl_matches_{now:%Y%m%dT%H%M%SZ}.json"
    if path.exists():
        raise FileExistsError(f"Snapshot already exists: {path}")
    url = fixtures_url()
    content = download_bytes(url, headers={"X-Auth-Token": api_key})
    return save_and_log(content, path, "current", url, now)


# --- Second source for quality checks (Wikipedia) ----------------------------


def wikipedia_titles(season: str) -> dict[str, str]:
    """Return {page kind: article title} for a season's Wikipedia match articles.

    Example: "2015-16" -> {"group_stage": "2015–16 UEFA Champions League group stage",
                           "knockout_phase": "2015–16 UEFA Champions League knockout phase",
                           "final": "2016 UEFA Champions League final"}
    The 36-team format (2024-25 on) has a "league phase" article instead of a group stage.
    The final has its own article, named after the year it was played.
    """
    build_season_url(season)  # validates the season string
    title_season = season.replace("-", "–")  # Wikipedia titles use an en dash
    first_phase = "league_phase" if int(season[:4]) >= 2024 else "group_stage"
    return {
        first_phase: f"{title_season} UEFA Champions League {first_phase.replace('_', ' ')}",
        "knockout_phase": f"{title_season} UEFA Champions League knockout phase",
        "final": f"{int(season[:4]) + 1} UEFA Champions League final",
    }


def wikipedia_api_url(title: str) -> str:
    """Return the API URL that fetches the current wikitext and revision ID of a page."""
    query = requests.models.PreparedRequest()
    query.prepare_url(WIKIPEDIA_API_URL, {
        "action": "query", "prop": "revisions", "rvprop": "ids|content", "rvslots": "main",
        "titles": title, "format": "json", "formatversion": "2", "redirects": "1",
    })
    return query.url


def collect_wikipedia_season(
    season: str,
    out_dir: Path = WIKIPEDIA_RAW_DIR,
    overwrite: bool = False,
    kinds: list[str] | None = None,
) -> list[Path]:
    """Save a season's Wikipedia API responses unchanged as JSON (one file per article).

    Args:
        kinds: Only these page kinds (e.g. ["league_phase"] for a season in progress,
            whose knockout and final articles may not exist yet). Default: all.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, title in wikipedia_titles(season).items():
        if kinds is not None and kind not in kinds:
            continue
        path = out_dir / f"{season}_{kind}.json"
        if not path.exists() or overwrite:
            url = wikipedia_api_url(title)
            content = download_bytes(url, headers={"User-Agent": HTTP_USER_AGENT})
            save_and_log(content, path, season, url, datetime.now(timezone.utc))
        paths.append(path)
    return paths


def collect_wikipedia_title_resolution(
    titles: list[str], out_path: Path, overwrite: bool = False
) -> Path:
    """Ask Wikipedia which final article each title leads to (redirects resolved).

    Saves the raw API responses (50 titles per request, the API limit) as one JSON
    list, logged like every other download.
    """
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    responses, urls = [], []
    for start in range(0, len(titles), 50):
        query = requests.models.PreparedRequest()
        query.prepare_url(WIKIPEDIA_API_URL, {
            "action": "query", "titles": "|".join(titles[start:start + 50]),
            "redirects": "1", "format": "json", "formatversion": "2",
        })
        urls.append(query.url)
        content = download_bytes(query.url, headers={"User-Agent": HTTP_USER_AGENT})
        responses.append(content.decode("utf-8"))
    content = ("[" + ",".join(responses) + "]").encode("utf-8")
    return save_and_log(content, out_path, "titles", " ".join(urls), datetime.now(timezone.utc))


def main() -> None:
    """Command-line entry point: collect historical seasons, fixtures or check pages."""
    parser = argparse.ArgumentParser(description="Download raw Champions League data.")
    parser.add_argument(
        "source", nargs="?", default="historical",
        choices=["historical", "fixtures", "wikipedia"],
    )
    args = parser.parse_args()

    if args.source == "historical":
        for saved in collect_seasons():
            print(f"{saved.name}: {saved.stat().st_size:,} bytes")
    elif args.source == "fixtures":
        saved = collect_fixtures(load_api_key())
        print(f"Saved snapshot {saved.name}: {saved.stat().st_size:,} bytes")
    else:
        for season in HISTORICAL_SEASONS:
            for saved in collect_wikipedia_season(season):
                print(f"{saved.name}: {saved.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
