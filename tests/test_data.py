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

import pytest

from src.data import collect

FAKE_CONTENT = "= test file\r\n  Ünïcode line\n".encode("utf-8")


@pytest.fixture
def fake_download(monkeypatch):
    """Replace download_bytes with a fake that records the URLs it was asked for."""
    calls: list[str] = []

    def _fake(url: str, timeout: int = 30) -> bytes:
        calls.append(url)
        return FAKE_CONTENT

    monkeypatch.setattr(collect, "download_bytes", _fake)
    return calls


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
    with (tmp_path / collect.LOG_FILENAME).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
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
