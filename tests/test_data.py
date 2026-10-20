"""Tests for src/data (collect.py now; clean.py and validate.py later).

Network calls are replaced with a fake download (monkeypatch) and files are written to
pytest's temporary folder (tmp_path), so tests never touch the internet or data/raw/.

Planned (Milestones 2-3), using small hand-built inputs, never downloaded data:
- team name normalisation maps spelling variants to one name
- dates are parsed and the table is sorted chronologically
- validation raises on missing columns, duplicate fixtures and invalid scores
"""

import csv
import hashlib
from datetime import datetime, timezone

import pytest

from src.data import collect

FAKE_CONTENT = "= test file\r\n  Ünïcode line\n".encode("utf-8")
FAKE_KEY = "test-key-not-real"
FIXED_TIME = datetime(2026, 10, 20, 17, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
def fake_download(monkeypatch):
    """Replace download_bytes with a fake that records each (url, headers) call."""
    calls: list[tuple[str, dict | None]] = []

    def _fake(url: str, headers: dict | None = None, timeout: int = 30) -> bytes:
        calls.append((url, headers))
        return FAKE_CONTENT

    monkeypatch.setattr(collect, "download_bytes", _fake)
    return calls


def read_log(folder):
    """Return the rows of a folder's download log as dicts."""
    with (folder / collect.LOG_FILENAME).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_build_season_url_uses_season_folder():
    url = collect.build_season_url("2025-26")
    assert url.endswith("/2025-26/cl.txt")


@pytest.mark.parametrize("bad", ["2025-2026", "2025/26", "25-26", "2025-27", ""])
def test_build_season_url_rejects_bad_seasons(bad):
    with pytest.raises(ValueError):
        collect.build_season_url(bad)


def test_collect_season_saves_bytes_unchanged(tmp_path, fake_download):
    path = collect.collect_season("2025-26", tmp_path)
    assert path.name == "2025-26_cl.txt"
    assert path.read_bytes() == FAKE_CONTENT  # no encoding or line-ending changes


def test_collect_season_writes_log_row(tmp_path, fake_download):
    collect.collect_season("2025-26", tmp_path)
    rows = read_log(tmp_path)
    assert len(rows) == 1
    assert rows[0]["season"] == "2025-26"
    assert rows[0]["bytes"] == str(len(FAKE_CONTENT))
    assert rows[0]["sha256"] == hashlib.sha256(FAKE_CONTENT).hexdigest()
    assert rows[0]["downloaded_at_utc"].endswith("+00:00")


def test_collect_season_skips_existing_file(tmp_path, fake_download):
    collect.collect_season("2025-26", tmp_path)
    collect.collect_season("2025-26", tmp_path)
    assert len(fake_download) == 1  # second call used the file on disk


def test_collect_season_overwrite_downloads_again(tmp_path, fake_download):
    collect.collect_season("2025-26", tmp_path)
    collect.collect_season("2025-26", tmp_path, overwrite=True)
    assert len(fake_download) == 2


def test_collect_seasons_returns_paths_in_order(tmp_path, fake_download):
    paths = collect.collect_seasons(["2011-12", "2012-13"], tmp_path)
    assert [p.name for p in paths] == ["2011-12_cl.txt", "2012-13_cl.txt"]


# --- football-data.org fixtures ---------------------------------------------


def test_load_api_key_reads_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(collect.FOOTBALL_DATA_KEY_VAR, f"  {FAKE_KEY}  ")
    assert collect.load_api_key(env_file=tmp_path / "missing.env") == FAKE_KEY


@pytest.mark.parametrize("value", [None, "", "   "])
def test_load_api_key_missing_raises_clear_error(tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(collect.FOOTBALL_DATA_KEY_VAR, raising=False)
    else:
        monkeypatch.setenv(collect.FOOTBALL_DATA_KEY_VAR, value)
    with pytest.raises(RuntimeError, match=collect.FOOTBALL_DATA_KEY_VAR):
        collect.load_api_key(env_file=tmp_path / "missing.env")


def test_collect_fixtures_saves_timestamped_snapshot(tmp_path, fake_download):
    path = collect.collect_fixtures(FAKE_KEY, tmp_path, now=FIXED_TIME)
    assert path.name == "cl_matches_20261020T170600Z.json"
    assert path.read_bytes() == FAKE_CONTENT


def test_collect_fixtures_sends_key_only_in_header(tmp_path, fake_download):
    path = collect.collect_fixtures(FAKE_KEY, tmp_path, now=FIXED_TIME)
    url, headers = fake_download[0]
    assert headers == {"X-Auth-Token": FAKE_KEY}
    assert FAKE_KEY not in url
    assert FAKE_KEY not in path.name
    assert FAKE_KEY not in (tmp_path / collect.LOG_FILENAME).read_text(encoding="utf-8")


def test_collect_fixtures_logs_current_season(tmp_path, fake_download):
    collect.collect_fixtures(FAKE_KEY, tmp_path, now=FIXED_TIME)
    row = read_log(tmp_path)[0]
    assert row["season"] == "current"
    assert row["downloaded_at_utc"] == "2026-10-20T17:06:00+00:00"


def test_collect_fixtures_keeps_every_snapshot(tmp_path, fake_download):
    later = FIXED_TIME.replace(hour=18)
    collect.collect_fixtures(FAKE_KEY, tmp_path, now=FIXED_TIME)
    collect.collect_fixtures(FAKE_KEY, tmp_path, now=later)
    assert len(list(tmp_path.glob("cl_matches_*.json"))) == 2
    assert len(read_log(tmp_path)) == 2


def test_collect_fixtures_never_overwrites_snapshot(tmp_path, fake_download):
    collect.collect_fixtures(FAKE_KEY, tmp_path, now=FIXED_TIME)
    with pytest.raises(FileExistsError):
        collect.collect_fixtures(FAKE_KEY, tmp_path, now=FIXED_TIME)
    assert len(fake_download) == 1  # no API call wasted on the refused snapshot
