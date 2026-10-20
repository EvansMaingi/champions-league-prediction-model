"""Tests for src/data: collect.py and the clean.py parser (validate.py later).

Network calls are replaced with a fake download (monkeypatch) and files are written to
pytest's temporary folder (tmp_path), so tests never touch the internet or data/raw/.
Parser tests use small hand-written Football.TXT snippets, never downloaded data.

Planned (Milestone 3):
- team name normalisation maps spelling variants to one name
- validation raises on missing columns, duplicate fixtures and invalid scores
"""

import csv
import hashlib
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.data import clean, collect

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


# --- clean.py: parsing Football.TXT -------------------------------------------

SAMPLE_SEASON = """= UEFA Champions League 2015/16

# Matches    4

▪ Gruppe G
  Tue Sep 15 2015
    20:45  Chelsea FC (ENG)        v Maccabi Tel Aviv (ISR)   4-0 (2-0)
           FC Porto (POR)          v Dynamo Kyiv (UKR)        2-2
▪ Finals, Round of 16
  Tue Mar 15 2016
    20:45  Atlético Madrid (ESP)   v PSV Eindhoven (NED)      8-7 pen. 0-0 a.e.t. (0-0)
  Wed Mar 16
    18:00  Bayern München (GER)    v Juventus (ITA)           4-2 a.e.t. (2-2, 0-2)
"""


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1-1 (0-0)", {"ft": (1, 1), "ht": (0, 0), "aet": (None, None), "pens": (None, None)}),
        ("0-0", {"ft": (0, 0), "ht": (None, None), "aet": (None, None), "pens": (None, None)}),
        ("4-1 a.e.t. (3-1, 1-0)", {"ft": (3, 1), "ht": (1, 0), "aet": (4, 1), "pens": (None, None)}),
        ("4-3 pen. 1-0 a.e.t. (1-0, 1-0)", {"ft": (1, 0), "ht": (1, 0), "aet": (1, 0), "pens": (4, 3)}),
        ("8-7 pen. 0-0 a.e.t. (0-0)", {"ft": (0, 0), "ht": (None, None), "aet": (0, 0), "pens": (8, 7)}),
    ],
)
def test_parse_score_formats(text, expected):
    parsed = clean.parse_score(text)
    for kind, (home, away) in expected.items():
        assert (parsed[f"home_{kind}"], parsed[f"away_{kind}"]) == (home, away)


@pytest.mark.parametrize("bad", ["1:1", "1-1 (0-0", "abandoned", "1-1 aet (1-1)"])
def test_parse_score_rejects_unknown_formats(bad):
    with pytest.raises(ValueError):
        clean.parse_score(bad)


def test_parse_date_line_carries_year_forward():
    first, year = clean.parse_date_line("Wed Sep 14 2011", None)
    second, year = clean.parse_date_line("Tue Sep 27", year)
    assert (first, second, year) == (date(2011, 9, 14), date(2011, 9, 27), 2011)


def test_parse_date_line_detects_wrong_weekday():
    with pytest.raises(ValueError, match="Weekday mismatch"):
        clean.parse_date_line("Mon Sep 14 2011", None)  # 14 Sep 2011 was a Wednesday


def test_parse_date_line_needs_a_year_first():
    with pytest.raises(ValueError):
        clean.parse_date_line("Tue Sep 27", None)


@pytest.mark.parametrize(
    "header, stage",
    [("Group A", "group"), ("Gruppe G", "group"), ("Group, Matchday 3", "group"),
     ("League, Matchday 8", "league"), ("Playoffs, Matchday 1", "playoff"),
     ("Round of 16", "round_of_16"), ("Finals, Quarterfinals", "quarterfinal"),
     ("Semifinals", "semifinal"), ("Final", "final"), ("Finals, Final", "final")],
)
def test_normalise_stage(header, stage):
    assert clean.normalise_stage(header) == stage


def test_parse_match_line_splits_teams_countries_and_score():
    row = clean.parse_match_line("21:00  Paris Saint-Germain FC (FRA) v FC Bayern München (GER)  5-4 (3-2)")
    assert row["kickoff_local"] == "21:00"
    assert (row["home_team"], row["home_country"]) == ("Paris Saint-Germain FC", "FRA")
    assert (row["away_team"], row["away_country"]) == ("FC Bayern München", "GER")
    assert (row["home_ft"], row["away_ft"], row["home_ht"], row["away_ht"]) == (5, 4, 3, 2)


def test_parse_season_text_builds_one_row_per_match():
    df = clean.parse_season_text(SAMPLE_SEASON, "2015-16", "sample.txt")
    assert len(df) == 4
    assert list(df.columns) == clean.COLUMNS
    assert df["stage"].tolist() == ["group", "group", "round_of_16", "round_of_16"]
    assert df["stage_raw"].iloc[0] == "Gruppe G"  # original header kept


def test_parse_season_text_infers_dates_and_inherits_kickoff_time():
    df = clean.parse_season_text(SAMPLE_SEASON, "2015-16")
    assert df["date"].dt.date.tolist() == [date(2015, 9, 15)] * 2 + [date(2016, 3, 15), date(2016, 3, 16)]
    assert df["kickoff_local"].tolist() == ["20:45", "20:45", "20:45", "18:00"]


def test_parse_season_text_keeps_missing_scores_missing():
    df = clean.parse_season_text(SAMPLE_SEASON, "2015-16")
    porto = df.iloc[1]
    assert pd.isna(porto["home_ht"]) and pd.isna(porto["home_aet"])  # unknown, not 0
    assert str(df["home_ht"].dtype) == "Int64"


def test_parse_season_text_fails_loudly_on_unknown_line():
    broken = SAMPLE_SEASON + "    Match postponed due to weather\n"
    with pytest.raises(ValueError, match=r"sample\.txt line 14"):
        clean.parse_season_text(broken, "2015-16", "sample.txt")


def test_parse_season_text_rejects_match_before_date():
    with pytest.raises(ValueError, match="before any stage or date"):
        clean.parse_season_text("▪ Group A\n  Chelsea FC (ENG) v Arsenal FC (ENG) 1-0\n", "2015-16")


def test_load_openfootball_sorts_seasons_chronologically(tmp_path):
    (tmp_path / "2016-17_cl.txt").write_text("▪ Group A\n  Tue Sep 13 2016\n    20:45  A FC (ENG) v B FC (ESP)  1-0\n", encoding="utf-8")
    (tmp_path / "2015-16_cl.txt").write_text(SAMPLE_SEASON, encoding="utf-8")
    df = clean.load_openfootball(tmp_path)
    assert df["date"].is_monotonic_increasing
    assert df["season"].iloc[-1] == "2016-17"


def test_load_openfootball_errors_when_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        clean.load_openfootball(tmp_path)
