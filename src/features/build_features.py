"""Create the target and pre-match features for every match.

Phase: Milestone 4 - Target definition (implemented), Milestone 5 - Feature engineering.

Previous stage: clean.py / validate.py produced the clean match table
(data/processed/matches.csv, loaded with clean.load_clean_matches).
Next stage: split.py divides this table chronologically.

Target (Milestone 4)
--------------------
`result` is the 90-minute outcome from the listed home side's point of view:
H (home win), D (draw), A (away win), stored as an ordered category H > D > A.
- 90 minutes, not after extra time: a pre-match 3-way prediction is about 90 minutes,
  and extra time is only played when the 90-minute score leaves a tie level. This
  changes 4 of 1,997 labels compared with using the score after extra time.
- Match level, not tie level: "who qualifies" is a different target (future extension).
- Neutral venues: H means "the team listed first won". Features must use the same
  orientation (home_* = listed first) so the label stays well-defined.
Note: scikit-learn sorts class labels alphabetically, so model.classes_ will be
['A', 'D', 'H']; always read predict_proba columns through classes_.

Leakage guard
-------------
Every column has a role in COLUMN_ROLES: "pre_match" (may become a feature),
"post_match" (known only after the final whistle; never a feature for that match) or
"identifier" (keys for sorting, joining and tracing; not features).
assert_valid_features(features, allowed) fails if a feature is post-match, or is not on
the explicit allowlist that Milestone 5 will declare.

TODO (Milestone 5, leakage rules):
- Rolling features use only EARLIER matches: sort chronologically and shift so a match
  never influences its own features; handle teams with little history explicitly.
- Post-match columns may enter features only through earlier matches
  (e.g. goals scored over the last FORM_WINDOW matches).
- For a second leg, the first-leg score is legitimate pre-match information.
- Before 2021-22 the away goals rule applied, so "level after leg 1" meant more than
  goal difference.
"""

import pandas as pd

TARGET = "result"
RESULT_CLASSES = ["H", "D", "A"]
RESULT_DTYPE = pd.CategoricalDtype(RESULT_CLASSES, ordered=True)

PRE_MATCH = [
    "season", "stage", "stage_raw", "home_team", "away_team", "home_country", "away_country",
    "neutral_venue", "neutral_reason", "played_abroad", "behind_closed_doors", "leg",
]
POST_MATCH = [
    TARGET,
    "home_ft", "away_ft", "home_ht", "away_ht", "home_aet", "away_aet", "home_pens", "away_pens",
    "ht_inferred",  # derived from the final score being 0-0
]
# Known only after a match, not (yet) in the table: forbidden so they can never be added
# as features by accident. ("Aggregate after this match" is post-match; the first-leg
# score going into a second leg is pre-match and will get its own name in Milestone 5.)
POST_MATCH_NOT_IN_TABLE = [
    "attendance", "went_to_extra_time", "decided_on_penalties", "tie_winner",
    "qualified", "home_qualified", "away_qualified", "aggregate_home", "aggregate_away",
]
IDENTIFIERS = [
    "date", "kickoff_utc",
    "kickoff_cet",  # for two suspended matches this is the restart time, known only later
    "tie_id", "source_file", "home_team_source", "away_team_source",
]

COLUMN_ROLES: dict[str, str] = {
    **{column: "pre_match" for column in PRE_MATCH},
    **{column: "post_match" for column in POST_MATCH + POST_MATCH_NOT_IN_TABLE},
    **{column: "identifier" for column in IDENTIFIERS},
}
FORBIDDEN_FEATURE_COLUMNS = frozenset(c for c, role in COLUMN_ROLES.items() if role == "post_match")


def add_result_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add `result` (ordered category H > D > A) from the 90-minute score.

    Extra time and penalties are ignored on purpose: a match level after 90 minutes is D
    even if it was decided later.

    Raises:
        ValueError: If any 90-minute score is missing, so no match gets an invented label.
    """
    missing = df["home_ft"].isna() | df["away_ft"].isna()
    if missing.any():
        raise ValueError(f"Missing 90-minute score: rows {df.index[missing].tolist()}")
    out = df.copy()
    home, away = out["home_ft"].astype(int), out["away_ft"].astype(int)
    labels = pd.Series("D", index=out.index)
    labels[home > away] = "H"
    labels[home < away] = "A"
    out[TARGET] = labels.astype(RESULT_DTYPE)
    return out


def unclassified_columns(columns) -> list[str]:
    """Columns that have no role in COLUMN_ROLES (every column must be classified)."""
    return [column for column in columns if column not in COLUMN_ROLES]


def assert_no_leakage(feature_columns: list[str]) -> None:
    """Raise if any feature is known only after the match (see POST_MATCH)."""
    leaked = sorted(set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS)
    if leaked:
        raise ValueError(f"Post-match columns cannot be features: {leaked}")


def assert_valid_features(feature_columns: list[str], allowed_columns: list[str]) -> None:
    """Raise if a feature is forbidden, or is not on the explicit allowlist.

    The allowlist cannot override the forbidden list: listing a post-match column as
    allowed is itself an error.
    """
    assert_no_leakage(feature_columns)
    allowed_but_forbidden = sorted(set(allowed_columns) & FORBIDDEN_FEATURE_COLUMNS)
    if allowed_but_forbidden:
        raise ValueError(f"Allowlist contains post-match columns: {allowed_but_forbidden}")
    not_allowed = sorted(set(feature_columns) - set(allowed_columns))
    if not_allowed:
        raise ValueError(f"Features not on the allowlist: {not_allowed}")


def result_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count and share of each class, in H, D, A order."""
    counts = df[TARGET].value_counts().reindex(RESULT_CLASSES)
    return pd.DataFrame({"matches": counts, "share": (counts / counts.sum()).round(4)})


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    from src.data.clean import load_clean_matches

    table = add_result_target(load_clean_matches(PROCESSED_DATA_DIR / "matches.csv"))
    print(result_counts(table).to_string())
