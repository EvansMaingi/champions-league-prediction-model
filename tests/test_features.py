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
    groups = [bf.PRE_MATCH, bf.POST_MATCH + bf.POST_MATCH_NOT_IN_TABLE, bf.IDENTIFIERS]
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


def test_assert_valid_features_accepts_allowed_pre_match_features():
    bf.assert_valid_features(["neutral_venue", "home_form_points"],
                             allowed_columns=["neutral_venue", "home_form_points", "away_form_points"])


def test_assert_valid_features_rejects_unlisted_feature():
    with pytest.raises(ValueError, match="not on the allowlist: \\['stage'\\]"):
        bf.assert_valid_features(["neutral_venue", "stage"], allowed_columns=["neutral_venue"])


def test_assert_valid_features_rejects_forbidden_even_if_allowed():
    with pytest.raises(ValueError, match="Post-match"):
        bf.assert_valid_features(["result"], allowed_columns=["result"])
    with pytest.raises(ValueError, match="Allowlist contains post-match"):
        bf.assert_valid_features(["neutral_venue"], allowed_columns=["neutral_venue", "home_aet"])
