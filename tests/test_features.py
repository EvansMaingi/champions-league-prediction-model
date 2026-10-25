"""Tests for src/features/build_features.py: the target and the leakage guard.

Inputs are small hand-built tables, plus one end-to-end table from the cleaning pipeline
to check that every column it produces has a role.
"""

import pandas as pd
import pytest

from src.data import clean
from src.features import build_features as bf


def scores(rows: list[tuple]) -> pd.DataFrame:
    """Table from (home_ft, away_ft, home_aet, away_aet) tuples; aet may be None."""
    df = pd.DataFrame(rows, columns=["home_ft", "away_ft", "home_aet", "away_aet"])
    return df.astype("Int64")


def test_result_labels_from_90_minute_scores():
    df = bf.add_result_target(scores([(2, 0, None, None), (1, 1, None, None), (0, 3, None, None)]))
    assert df["result"].tolist() == ["H", "D", "A"]


def test_result_ignores_extra_time_and_penalties():
    # The 4 real cases where the label depends on the choice (90 minutes -> after extra time):
    df = bf.add_result_target(scores([
        (1, 1, 4, 1),   # 2014 final: D, not H
        (2, 2, 4, 2),   # 2016 Bayern v Juventus: D, not H
        (1, 2, 4, 2),   # 2017 Real Madrid v Bayern: A, not H
        (1, 0, 2, 3),   # 2020 Liverpool v Atlético: H, not A
    ]))
    assert df["result"].tolist() == ["D", "D", "A", "H"]


def test_result_is_ordered_category():
    df = bf.add_result_target(scores([(1, 0, None, None)]))
    assert df["result"].dtype == bf.RESULT_DTYPE
    assert list(df["result"].cat.categories) == ["H", "D", "A"] and df["result"].cat.ordered


def test_result_never_missing_and_missing_scores_raise():
    df = bf.add_result_target(scores([(0, 0, None, None), (5, 4, None, None)]))
    assert df["result"].notna().all()
    with pytest.raises(ValueError, match="Missing 90-minute score"):
        bf.add_result_target(scores([(1, None, None, None)]))


def test_add_result_target_does_not_modify_input():
    original = scores([(1, 0, None, None)])
    bf.add_result_target(original)
    assert "result" not in original.columns


# --- Column roles -----------------------------------------------------------------

def test_every_pipeline_column_has_a_role(tmp_path):
    """If cleaning ever adds a column, this fails until the column is classified."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2015-16_cl.txt").write_text(
        "▪ Final\n  Sat May 28 2016\n    20:45  Real Madrid (ESP) v Atlético Madrid (ESP)  5-3 pen. 1-1 a.e.t. (1-1, 1-0)\n",
        encoding="utf-8")
    reference = tmp_path / "reference"
    reference.mkdir()
    pd.DataFrame([
        ["openfootball", "", "Real Madrid", "ESP", "Real Madrid", "Real Madrid CF", "test"],
        ["openfootball", "", "Atlético Madrid", "ESP", "Atlético Madrid", "Atlético Madrid", "test"],
    ], columns=["source", "source_id", "source_name", "country", "standard_name", "wikipedia_article",
                "evidence"]).to_csv(reference / "team_names.csv", index=False)
    (reference / "match_context.csv").write_text(
        "season,date,home_team,away_team,neutral_venue,neutral_reason,played_abroad,behind_closed_doors\n",
        encoding="utf-8")
    table = bf.add_result_target(clean.clean_matches(raw, reference))
    assert bf.unclassified_columns(table.columns) == []
    assert table["result"].tolist() == ["D"]  # level after 90 minutes, decided on penalties


def test_roles_do_not_overlap():
    groups = [bf.PRE_MATCH, bf.DERIVED_PRE_MATCH, bf.POST_MATCH + bf.POST_MATCH_NOT_IN_TABLE, bf.IDENTIFIERS]
    assert sum(len(g) for g in groups) == len(bf.COLUMN_ROLES)


@pytest.mark.parametrize("column", [
    "result", "home_ft", "away_ft", "home_ht", "away_ht", "home_aet", "away_aet",
    "home_pens", "away_pens", "ht_inferred", "attendance", "went_to_extra_time",
    "decided_on_penalties", "tie_winner", "qualified", "home_qualified", "away_qualified",
])
def test_post_match_columns_are_forbidden(column):
    assert column in bf.FORBIDDEN_FEATURE_COLUMNS


def test_pre_match_context_is_not_forbidden():
    for column in ("neutral_venue", "played_abroad", "behind_closed_doors", "leg", "stage"):
        assert column not in bf.FORBIDDEN_FEATURE_COLUMNS


# --- Leakage guard ------------------------------------------------------------------

def test_assert_no_leakage_rejects_target_and_scores():
    with pytest.raises(ValueError, match=r"\['home_ft', 'result'\]"):
        bf.assert_no_leakage(["neutral_venue", "result", "home_ft"])
    bf.assert_no_leakage(["neutral_venue", "home_form_points"])  # no error


def roles_with_derived(*names: str) -> dict[str, str]:
    """COLUMN_ROLES plus some registered derived pre-match features (for tests)."""
    return {**bf.COLUMN_ROLES, **{name: "derived_pre_match" for name in names}}


def test_assert_valid_features_accepts_allowed_pre_match_features():
    bf.assert_valid_features(["neutral_venue", "stage"], allowed_columns=["neutral_venue", "stage", "leg"])


def test_assert_valid_features_accepts_registered_derived_feature():
    roles = roles_with_derived("home_form_points", "away_form_points")
    bf.assert_valid_features(["neutral_venue", "home_form_points"],
                             allowed_columns=["neutral_venue", "home_form_points", "away_form_points"],
                             roles=roles)


def test_assert_valid_features_rejects_unlisted_feature():
    with pytest.raises(ValueError, match="not on the allowlist: \\['stage'\\]"):
        bf.assert_valid_features(["neutral_venue", "stage"], allowed_columns=["neutral_venue"])


def test_assert_valid_features_rejects_forbidden_even_if_allowed():
    with pytest.raises(ValueError, match="Features contains post-match"):
        bf.assert_valid_features(["result"], allowed_columns=["result"])
    with pytest.raises(ValueError, match="Allowlist contains post-match"):
        bf.assert_valid_features(["neutral_venue"], allowed_columns=["neutral_venue", "home_aet"])


def test_unclassified_derived_name_fails_even_if_allowlisted():
    """The gap from the Milestone 4 review: a new name with no role must not pass."""
    assert "home_led_at_halftime" not in bf.COLUMN_ROLES
    with pytest.raises(ValueError, match="without a role.*home_led_at_halftime"):
        bf.assert_valid_features(["home_led_at_halftime"], allowed_columns=["home_led_at_halftime"])


def test_unclassified_name_on_allowlist_alone_fails():
    with pytest.raises(ValueError, match="Allowlist contains columns without a role"):
        bf.assert_valid_features(["neutral_venue"], allowed_columns=["neutral_venue", "mystery_column"])


def test_identifier_cannot_be_a_feature():
    with pytest.raises(ValueError, match="not pre-match features: \\['date'\\]"):
        bf.assert_valid_features(["date"], allowed_columns=["date"])


# --- Part A: team match log and recent form -----------------------------------------

def matches(rows: list[tuple]) -> pd.DataFrame:
    """Match table from (season, date, home, away, home_ft, away_ft); kickoff 20:00 UTC."""
    df = pd.DataFrame(rows, columns=["season", "date", "home_team", "away_team", "home_ft", "away_ft"])
    df["date"] = pd.to_datetime(df["date"])
    df["kickoff_utc"] = (df["date"] + pd.Timedelta(hours=20)).dt.tz_localize("UTC")
    df[["home_ft", "away_ft"]] = df[["home_ft", "away_ft"]].astype("Int64")
    return df


# Team A plays five matches; B, C, D are opponents. A's points: 3, 0, 1, 3, 3.
SEQUENCE = matches([
    ("2020-21", "2020-10-01", "A", "B", 2, 0),
    ("2020-21", "2020-10-08", "C", "A", 1, 0),
    ("2020-21", "2020-10-15", "A", "D", 1, 1),
    ("2021-22", "2021-09-14", "B", "A", 0, 3),
    ("2021-22", "2021-09-21", "A", "C", 4, 2),
])


def team_rows(form: pd.DataFrame, team: str) -> pd.DataFrame:
    return form[form["team"] == team].reset_index(drop=True)


def test_team_match_log_has_two_rows_per_match_from_each_perspective():
    log = bf.team_match_log(SEQUENCE)
    assert len(log) == 2 * len(SEQUENCE)
    first = log[log["match_index"] == 0].set_index("team")
    assert (first.loc["A", "goals_for"], first.loc["A", "goals_against"], first.loc["A", "points"]) == (2, 0, 3)
    assert (first.loc["B", "goals_for"], first.loc["B", "goals_against"], first.loc["B", "points"]) == (0, 2, 0)
    assert bool(first.loc["A", "is_home"]) and not bool(first.loc["B", "is_home"])


def test_rolling_form_uses_only_earlier_matches():
    a = team_rows(bf.add_rolling_form(bf.team_match_log(SEQUENCE)), "A")
    assert a["form_points"].isna().iloc[0]                    # first match: no history
    assert a["form_points"].iloc[1:].round(3).tolist() == [3.0, 1.5, 1.333, 1.75]
    assert a["form_n"].tolist() == [0, 1, 2, 3, 4]
    assert a["form_goals_for"].iloc[4] == pytest.approx((2 + 0 + 1 + 3) / 4)


def test_rolling_form_window_drops_the_oldest_match():
    a = team_rows(bf.add_rolling_form(bf.team_match_log(SEQUENCE), window=2), "A")
    assert a["form_points"].iloc[4] == pytest.approx((1 + 3) / 2)   # only matches 3 and 4
    assert a["form_n"].max() == 2


def test_rolling_form_never_mixes_teams():
    c = team_rows(bf.add_rolling_form(bf.team_match_log(SEQUENCE)), "C")
    assert c["form_points"].isna().iloc[0] and c["form_points"].iloc[1] == 3.0  # C's own win only


def test_season_scope_and_form_n_season():
    carry = team_rows(bf.add_rolling_form(bf.team_match_log(SEQUENCE), scope="carry_over"), "A")
    reset = team_rows(bf.add_rolling_form(bf.team_match_log(SEQUENCE), scope="season_reset"), "A")
    assert carry["form_n"].iloc[3] == 3 and carry["form_n_season"].iloc[3] == 0   # all from last season
    assert carry["form_n_season"].iloc[4] == 1
    assert pd.isna(reset["form_points"].iloc[3]) and reset["form_n"].iloc[3] == 0  # new season, no history


def test_cold_start_gives_missing_values_and_flag():
    df = bf.add_form_features(SEQUENCE)
    assert pd.isna(df.loc[0, "home_form_points"]) and not df.loc[0, "home_has_history"]
    assert df.loc[2, "home_has_history"] and not df.loc[2, "away_has_history"]   # D's first match


def test_features_unchanged_when_later_matches_are_removed():
    """Truncation test: catches leakage from LATER matches."""
    full = bf.add_form_features(SEQUENCE)
    cut = bf.add_form_features(SEQUENCE[SEQUENCE["date"] < "2021-09-15"])
    columns = [c for c in bf.FORM_FEATURES]
    pd.testing.assert_frame_equal(full.loc[cut.index, columns], cut[columns])


def test_changing_a_score_only_affects_later_matches():
    """Perturbation test: catches a match leaking ITS OWN result (a missing shift)."""
    columns = [c for c in bf.FORM_FEATURES]
    base = bf.add_form_features(SEQUENCE)
    changed = SEQUENCE.copy()
    changed.loc[2, ["home_ft", "away_ft"]] = [9, 0]
    after = bf.add_form_features(changed)
    pd.testing.assert_frame_equal(base.loc[:2, columns], after.loc[:2, columns])  # itself and earlier
    assert not base.loc[3:, columns].equals(after.loc[3:, columns])                  # later matches see it


def test_hand_worked_bayern_example():
    """Bayern's real last five before PSG v Bayern, 28 April 2026 (semi-final, first leg)."""
    bayern = matches([
        ("2025-26", "2026-01-28", "PSV Eindhoven", "Bayern Munich", 1, 2),
        ("2025-26", "2026-03-10", "Atalanta", "Bayern Munich", 1, 6),
        ("2025-26", "2026-03-18", "Bayern Munich", "Atalanta", 4, 1),
        ("2025-26", "2026-04-07", "Real Madrid", "Bayern Munich", 1, 2),
        ("2025-26", "2026-04-15", "Bayern Munich", "Real Madrid", 4, 3),
        ("2025-26", "2026-04-28", "Paris Saint-Germain", "Bayern Munich", 5, 4),
    ])
    row = bf.add_form_features(bayern).iloc[-1]
    assert row["away_form_points"] == 3.0
    assert row["away_form_goals_for"] == pytest.approx(18 / 5)
    assert row["away_form_goals_against"] == pytest.approx(7 / 5)
    assert (row["away_form_n"], row["away_form_n_season"]) == (5, 5)


def test_part_a_features_are_registered_and_allowed():
    bf.assert_valid_features(bf.FORM_FEATURES, bf.ALLOWED_FEATURES)
    assert set(bf.FORM_FEATURES) <= set(bf.DERIVED_PRE_MATCH)
    produced = [c for c in bf.add_form_features(SEQUENCE).columns if c.startswith(("home_form", "away_form")) or c.endswith("has_history")]
    assert sorted(produced) == sorted(bf.FORM_FEATURES)


# --- Part B: Elo rating -----------------------------------------------------------------

def test_elo_expected_score_basics():
    assert bf.elo_expected(1500, 1500) == 0.5
    assert bf.elo_expected(1700, 1500) == pytest.approx(1 / (1 + 10 ** (-0.5)))  # 200 points: ~0.76
    assert bf.elo_expected(1600, 1500) + bf.elo_expected(1500, 1600) == pytest.approx(1.0)
    assert bf.elo_expected(1500, 1500, home_advantage=60) > 0.5


def test_elo_records_rating_before_the_match_and_is_zero_sum():
    df = bf.add_elo_features(SEQUENCE.iloc[:2])
    assert (df.loc[0, "home_elo"], df.loc[0, "away_elo"]) == (1500.0, 1500.0)   # before any update
    # Match 0: A (home, HFA 60) beat B. A gains K * (1 - E), B loses the same.
    gain = bf.ELO_K * (1 - bf.elo_expected(1500, 1500, bf.ELO_HOME_ADVANTAGE))
    assert df.loc[1, "away_elo"] == pytest.approx(1500 + gain)                    # A's rating before match 1
    assert df.loc[0, "elo_diff"] == 0.0


def test_elo_hand_worked_two_matches():
    two = matches([("2020-21", "2020-10-01", "A", "B", 1, 0), ("2020-21", "2020-10-08", "B", "A", 1, 1)])
    e1 = 1 / (1 + 10 ** ((1500 - 1560) / 400))                 # A at home: E = 0.5855
    a, b = 1500 + 20 * (1 - e1), 1500 - 20 * (1 - e1)          # 1508.29, 1491.71
    row = bf.add_elo_features(two).iloc[1]
    assert (row["home_elo"], row["away_elo"]) == (pytest.approx(b), pytest.approx(a))
    assert row["elo_diff"] == pytest.approx(b - a)


def test_elo_no_home_advantage_at_neutral_or_abroad_venues():
    one = matches([("2020-21", "2020-10-01", "A", "B", 1, 0), ("2020-21", "2020-10-08", "A", "C", 0, 0)])
    neutral = one.assign(neutral_venue=[True, False], played_abroad=False)
    gain_home = bf.add_elo_features(one).loc[1, "home_elo"] - 1500
    gain_neutral = bf.add_elo_features(neutral).loc[1, "home_elo"] - 1500
    assert gain_neutral == pytest.approx(20 * 0.5)               # E = 0.5 without home advantage
    assert gain_home < gain_neutral                              # a home win was more expected


def test_elo_regresses_toward_start_between_seasons():
    two = matches([("2020-21", "2020-10-01", "A", "B", 1, 0), ("2021-22", "2021-09-14", "A", "B", 0, 0)])
    after_first = 1500 + 20 * (1 - bf.elo_expected(1500, 1500, 60))
    row = bf.add_elo_features(two).iloc[1]
    assert row["home_elo"] == pytest.approx(after_first - (after_first - 1500) / 3)


def test_elo_counts_rated_matches():
    df = bf.add_elo_features(SEQUENCE)
    assert df["home_elo_n"].tolist() == [0, 0, 2, 1, 4]   # A, C, A, B, A before each match
    assert df["away_elo_n"].tolist() == [0, 1, 0, 3, 1]


def test_elo_unchanged_when_later_matches_are_removed():
    full = bf.add_elo_features(SEQUENCE)
    cut = bf.add_elo_features(SEQUENCE[SEQUENCE["date"] < "2021-09-15"])
    pd.testing.assert_frame_equal(full.loc[cut.index, bf.ELO_FEATURES], cut[bf.ELO_FEATURES])


def test_elo_changing_a_score_only_affects_later_matches():
    base = bf.add_elo_features(SEQUENCE)
    changed = SEQUENCE.copy()
    changed.loc[2, ["home_ft", "away_ft"]] = [0, 5]
    after = bf.add_elo_features(changed)
    pd.testing.assert_frame_equal(base.loc[:2, bf.ELO_FEATURES], after.loc[:2, bf.ELO_FEATURES])
    assert not base.loc[3:, bf.ELO_FEATURES].equals(after.loc[3:, bf.ELO_FEATURES])


def test_part_b_features_are_registered_and_allowed():
    bf.assert_valid_features(bf.ELO_FEATURES, bf.ALLOWED_FEATURES)
    produced = set(bf.add_elo_features(SEQUENCE).columns) - set(SEQUENCE.columns)
    assert produced == set(bf.ELO_FEATURES)
