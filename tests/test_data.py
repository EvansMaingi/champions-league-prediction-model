"""Tests for src/data: collect.py, clean.py and validate.py.

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

from src.data import clean, collect, validate

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


# --- Wikipedia pages (second source) --------------------------------------------


def test_wikipedia_titles_per_format():
    assert collect.wikipedia_titles("2015-16") == {
        "group_stage": "2015–16 UEFA Champions League group stage",
        "knockout_phase": "2015–16 UEFA Champions League knockout phase",
        "final": "2016 UEFA Champions League final",
    }
    assert "league_phase" in collect.wikipedia_titles("2024-25")


def test_collect_wikipedia_season_sends_user_agent_and_filters_kinds(tmp_path, fake_download):
    paths = collect.collect_wikipedia_season("2026-27", tmp_path, kinds=["league_phase"])
    assert [p.name for p in paths] == ["2026-27_league_phase.json"]
    url, headers = fake_download[0]
    assert "league+phase" in url
    assert headers["User-Agent"].startswith("champions-league-prediction-model")


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
    assert row["kickoff_cet"] == "21:00"
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
    assert df["kickoff_cet"].tolist() == ["20:45", "20:45", "20:45", "18:00"]


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


# --- clean.py: cleaning steps -------------------------------------------------

KNOCKOUT_SEASON = """▪ Round of 16
  Tue Feb 16 2016
    20:45  Paris Saint-Germain (FRA) v Chelsea FC (ENG)    2-1 (1-1)
  Tue Mar 1
    20:45  Lyon (FRA)              v Porto (POR)            0-0
  Wed Mar 9
    20:45  Chelsea FC (ENG)        v Paris Saint-Germain (FRA)  1-2 (1-1)
  Tue Mar 15
    20:45  Porto (POR)             v Lyon (FRA)             3-2 pen. 0-0 a.e.t. (0-0)
▪ Final
  Sat May 28
    20:45  Paris Saint-Germain (FRA) v Porto (POR)          1-1 (0-1)
"""


def knockout_table() -> pd.DataFrame:
    """Small parsed knockout season: two two-legged ties and a final."""
    return clean.parse_season_text(KNOCKOUT_SEASON, "2015-16")


def test_add_kickoff_utc_handles_winter_and_summer_time():
    df = clean.add_kickoff_utc(knockout_table())
    utc = df["kickoff_utc"].dt.strftime("%H:%M").tolist()
    assert utc[0] == "19:45"   # 16 Feb: CET = UTC+1
    assert utc[-1] == "18:45"  # 28 May: CEST = UTC+2
    assert str(df["kickoff_utc"].dt.tz) == "UTC"


def test_add_ht_inferred_fills_only_goalless_matches():
    df = clean.add_ht_inferred(knockout_table())
    goalless = (df["home_ft"] == 0) & (df["away_ft"] == 0)  # Lyon v Porto, both legs
    assert goalless.sum() == 2
    assert (df.loc[goalless, ["home_ht", "away_ht"]] == 0).all().all()
    assert df["ht_inferred"].tolist() == goalless.tolist()


def test_add_ht_inferred_leaves_other_missing_values_missing():
    df = knockout_table()
    df.loc[0, ["home_ht", "away_ht"]] = pd.NA  # 2-1 at 90 minutes: half-time unknown
    out = clean.add_ht_inferred(df)
    assert pd.isna(out.loc[0, "home_ht"]) and not out.loc[0, "ht_inferred"]


def context_rows(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=["season", "date", "home_team", "away_team", "neutral_venue",
                                        "neutral_reason", "played_abroad", "behind_closed_doors"])
    table["played_abroad"] = table["played_abroad"].fillna(False).astype(bool)
    return table


def test_add_match_context_flags_listed_matches_only():
    context = context_rows([
        {"season": "2015-16", "date": pd.Timestamp("2016-05-28"), "home_team": "Paris Saint-Germain",
         "away_team": "Porto", "neutral_venue": True, "neutral_reason": "final", "behind_closed_doors": False},
        {"season": "2015-16", "date": pd.Timestamp("2016-03-01"), "home_team": "Lyon",
         "away_team": "Porto", "neutral_venue": False, "neutral_reason": "", "behind_closed_doors": True},
    ])
    df = clean.add_match_context(knockout_table(), context)
    assert df["neutral_venue"].sum() == 1 and df["behind_closed_doors"].sum() == 1
    assert df.loc[df["neutral_venue"], "neutral_reason"].item() == "final"
    assert set(df.loc[~df["neutral_venue"], "neutral_reason"]) == {""}
    assert not df["played_abroad"].any() and df["played_abroad"].dtype == bool


def test_add_match_context_keeps_played_abroad_separate_from_neutral():
    context = context_rows([
        {"season": "2015-16", "date": pd.Timestamp("2016-02-16"), "home_team": "Paris Saint-Germain",
         "away_team": "Chelsea FC", "neutral_venue": False, "neutral_reason": "",
         "played_abroad": True, "behind_closed_doors": False},
    ])
    df = clean.add_match_context(knockout_table(), context)
    assert df["played_abroad"].sum() == 1 and df["neutral_venue"].sum() == 0


def test_add_match_context_rejects_rows_that_match_nothing():
    context = context_rows([
        {"season": "2015-16", "date": pd.Timestamp("2016-05-29"), "home_team": "Paris Saint-Germain",
         "away_team": "Porto", "neutral_venue": True, "neutral_reason": "final", "behind_closed_doors": False},
    ])
    with pytest.raises(ValueError, match="match no match"):
        clean.add_match_context(knockout_table(), context)


def test_add_ties_numbers_legs_and_singles():
    df = clean.add_ties(knockout_table())
    assert df["leg"].tolist() == ["1", "1", "2", "2", "single"]
    assert df["tie_id"].tolist() == ["2015-16_round_of_16_01", "2015-16_round_of_16_02",
                                     "2015-16_round_of_16_01", "2015-16_round_of_16_02", "2015-16_final_01"]


def test_add_ties_leaves_group_matches_without_tie():
    parsed = clean.parse_season_text(SAMPLE_SEASON, "2015-16")
    df = clean.add_ties(parsed[parsed["stage"] == "group"])
    assert df[["tie_id", "leg"]].isna().all().all()


def test_add_ties_rejects_lone_match_in_two_legged_stage():
    lone = knockout_table().drop(index=2)  # remove PSG v Chelsea's second leg
    with pytest.raises(ValueError, match="lone match"):
        clean.add_ties(lone)


def test_add_ties_allows_single_matches_in_lisbon_final_8():
    text = "▪ Quarterfinals\n  Wed Aug 12 2020\n    21:00  Atalanta (ITA) v Paris Saint-Germain (FRA)  1-2 (1-0)\n"
    df = clean.add_ties(clean.parse_season_text(text, "2019-20"))
    assert df["leg"].tolist() == ["single"]


def test_add_ties_rejects_second_leg_without_home_away_swap():
    df = knockout_table()
    df.loc[2, ["home_team", "away_team"]] = ["Paris Saint-Germain", "Chelsea FC"]
    with pytest.raises(ValueError, match="does not swap"):
        clean.add_ties(df)


def test_normalise_country_codes_maps_monaco_to_france():
    df = knockout_table()
    df.loc[0, "home_country"] = "MCO"
    out = clean.normalise_country_codes(df)
    assert out.loc[0, "home_country"] == "FRA"
    assert out["away_country"].tolist() == df["away_country"].tolist()  # others unchanged


def test_normalise_team_names_maps_and_keeps_source_names():
    mapping = {"Paris Saint-Germain": "Paris Saint-Germain", "Chelsea FC": "Chelsea",
               "Lyon": "Lyon", "Porto": "Porto"}
    df = clean.normalise_team_names(knockout_table(), mapping)
    assert "Chelsea" in set(df["away_team"]) and "Chelsea FC" not in set(df["away_team"])
    assert "Chelsea FC" in set(df["away_team_source"])


def test_normalise_team_names_fails_on_unmapped_name():
    with pytest.raises(ValueError, match="Chelsea FC"):
        clean.normalise_team_names(knockout_table(), {"Paris Saint-Germain": "PSG", "Lyon": "Lyon", "Porto": "Porto"})


# --- clean.py: full pipeline -----------------------------------------------------


def write_reference(folder, team_rows):
    """Write a minimal reference folder: team_names.csv and an empty match_context.csv."""
    folder.mkdir()
    pd.DataFrame(team_rows, columns=["source", "source_id", "source_name", "country", "standard_name",
                                     "wikipedia_article", "evidence"]).to_csv(folder / "team_names.csv", index=False)
    (folder / "match_context.csv").write_text(
        "season,date,home_team,away_team,neutral_venue,neutral_reason,played_abroad,behind_closed_doors\n",
        encoding="utf-8")


TEAM_ROWS = [
    ["openfootball", "", "Paris Saint-Germain", "FRA", "Paris Saint-Germain", "Paris Saint-Germain FC", "test"],
    ["openfootball", "", "Chelsea FC", "ENG", "Chelsea", "Chelsea F.C.", "test"],
    ["openfootball", "", "Lyon", "FRA", "Olympique Lyonnais", "Olympique Lyonnais", "test"],
    ["openfootball", "", "Porto", "POR", "Porto", "FC Porto", "test"],
]


def test_clean_matches_end_to_end_saves_and_round_trips(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2015-16_cl.txt").write_text(KNOCKOUT_SEASON, encoding="utf-8")
    write_reference(tmp_path / "reference", TEAM_ROWS)

    df = clean.clean_matches(raw, tmp_path / "reference", out_dir=tmp_path / "processed")
    assert len(df) == 5 and set(df["home_team"]) == {"Paris Saint-Germain", "Chelsea", "Olympique Lyonnais", "Porto"}
    assert df["ht_inferred"].sum() == 2 and df["tie_id"].nunique() == 3

    saved = clean.load_clean_matches(tmp_path / "processed" / clean.PROCESSED_FILE)
    validate.validate_matches(saved)
    assert saved["home_ft"].dtype == "Int64" and saved["neutral_venue"].dtype == bool
    assert saved["leg"].tolist() == df["leg"].tolist()


def test_clean_matches_stops_on_team_table_without_country(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2015-16_cl.txt").write_text(KNOCKOUT_SEASON, encoding="utf-8")
    rows = [row.copy() for row in TEAM_ROWS]
    rows[1][3] = ""  # Chelsea without a country
    write_reference(tmp_path / "reference", rows)
    with pytest.raises(ValueError, match="rows without country"):
        clean.clean_matches(raw, tmp_path / "reference")


# --- validate.py ----------------------------------------------------------------


def team_table(rows):
    return pd.DataFrame(rows, columns=["source_name", "country", "standard_name", "wikipedia_article"])


def test_validate_team_table_accepts_consistent_table():
    validate.validate_team_table(team_table([["Bayern München", "GER", "Bayern Munich", "FC Bayern Munich"],
                                             ["FC Bayern München", "GER", "Bayern Munich", "FC Bayern Munich"]]))


@pytest.mark.parametrize("rows, message", [
    ([["Como 1907", "", "Como", "Como 1907"]], "rows without country"),
    ([["A", "GER", "Club", "Club A"], ["B", "AUT", "Club", "Club A"]], "more than one country"),
    ([["A", "GER", "Club", "Club A"], ["B", "GER", "Club", "Club B"]], "more than one wikipedia_article"),
    ([["A", "GER", "Club A", "Article"], ["B", "GER", "Club B", "Article"]], "more than one standard_name"),
])
def test_validate_team_table_rejects_problems(rows, message):
    with pytest.raises(ValueError, match=message):
        validate.validate_team_table(team_table(rows))


def test_validate_matches_rejects_missing_column_duplicates_and_unsorted():
    df = clean.add_ties(clean.add_match_context(clean.add_ht_inferred(clean.add_kickoff_utc(knockout_table())),
                                                context_rows([])))
    validate.validate_matches(df)  # passes
    with pytest.raises(ValueError, match="Missing columns"):
        validate.validate_matches(df.drop(columns="leg"))
    with pytest.raises(ValueError, match="Duplicate"):
        validate.check_no_duplicate_matches(pd.concat([df, df.iloc[[0]]]))
    with pytest.raises(ValueError, match="chronologically"):
        validate.check_chronological(df.iloc[::-1])



def test_check_teams_mapped_lists_every_missing_name():
    with pytest.raises(ValueError, match=r"2 team name\(s\).*Chelsea FC.*Lyon"):
        validate.check_teams_mapped(knockout_table(), {"Paris Saint-Germain": "PSG", "Porto": "Porto"})


def test_check_scores_consistent_passes_real_formats():
    validate.check_scores_consistent(knockout_table())  # no error


def test_check_scores_consistent_rejects_half_time_above_full_time():
    df = knockout_table()
    df.loc[0, "home_ht"] = 3  # 3 at half-time but 2 at full time
    with pytest.raises(ValueError, match="half-time > 90 minutes"):
        validate.check_scores_consistent(df)


def test_check_shootouts_accepts_level_aggregate_in_second_leg():
    validate.check_shootouts(clean.add_ties(knockout_table()))  # 0-0, then 0-0 a.e.t.: level


def test_check_shootouts_accepts_level_two_legs_with_unlevel_second_leg():
    df = clean.add_ties(knockout_table())
    df.loc[df["home_team"].eq("Lyon"), "home_ft"] = 1                       # leg 1: Lyon 1-0 Porto
    df.loc[df["home_team"].eq("Porto") & df["stage"].eq("round_of_16"), "home_aet"] = 1  # leg 2: Porto 1-0 a.e.t.
    validate.check_shootouts(df)  # aggregate 1-1, so the shootout is valid


def test_check_shootouts_rejects_unlevel_aggregate():
    df = clean.add_ties(knockout_table())
    df.loc[df["home_team"].eq("Lyon"), "home_ft"] = 1  # leg 1: Lyon 1-0 Porto; leg 2 0-0: aggregate 1-0
    with pytest.raises(ValueError, match="not level"):
        validate.check_shootouts(df)
