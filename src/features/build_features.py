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
"derived_pre_match" (a feature built in this module, registered in DERIVED_PRE_MATCH),
"post_match" (known only after the final whistle; never a feature for that match) or
"identifier" (keys for sorting, joining and tracing; not features).
assert_valid_features(features, allowed) is the gate for model inputs: a feature must be
on the allowlist that Milestone 5 will declare AND have an explicit pre-match role.
A column without a role fails, so a new derived name cannot slip through unclassified.

Features (Milestone 5)
----------------------
Part A, recent form (add_form_features): from a long per-team log, each team's mean
points, goals scored and goals conceded over its previous FORM_WINDOW CL matches, via
shift(1) + rolling, so a match never sees its own result. form_n counts the matches the
window used, form_n_season how many are from the current season; no history gives
missing values and *_has_history = False. Two tests guard this: removing later matches
must not change earlier features (catches leaks from the future), and changing a
match's score must not change its own or earlier features (catches a missing shift).
Note for Milestone 6: the data starts in 2011-12, so clubs' first matches that season
look like cold starts although they had earlier CL history (left-censoring).

Part B, strength rating (add_elo_features): a Champions-League-only Elo computed in
kickoff order; the feature is each side's rating BEFORE the match (recorded before the
update, the Elo equivalent of shift(1)). Unlike form, it accounts for opponent
strength. Start 1500, K 20, home advantage 60 (0 at neutral or abroad venues), and 1/3
regression toward 1500 between seasons, all fixed in advance; Milestone 7 may tune them
on validation data only. Known limits (docs/data_notes.md): the rating scale needs about
four seasons to spread out, and newcomers starting at 1500 are overrated.

TODO (Part C):
- Context features; for a second leg, the first-leg score is legitimate
  pre-match information (before 2021-22 the away goals rule applied, so "level after
  leg 1" meant more than goal difference).
"""

from typing import Literal

import pandas as pd

from src.config import FORM_WINDOW

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

# Features created in this module (Milestone 5) are registered here, one entry per
# column, next to the code that builds them. Registering a name declares that it is
# computed only from information available before kickoff.
# Part A (add_form_features): form over each team's previous FORM_WINDOW CL matches.
FORM_FEATURES: list[str] = [
    f"{side}_{name}"
    for side in ("home", "away")
    for name in ("form_points", "form_goals_for", "form_goals_against", "form_n", "form_n_season", "has_history")
]
# Part B (add_elo_features): Champions-League-only Elo, the rating BEFORE kickoff.
ELO_FEATURES: list[str] = ["home_elo", "away_elo", "elo_diff", "home_elo_n", "away_elo_n"]
DERIVED_PRE_MATCH: list[str] = [*FORM_FEATURES, *ELO_FEATURES]

# Allowlist of model inputs, declared part by part (Milestone 5).
ALLOWED_FEATURES: list[str] = [*FORM_FEATURES, *ELO_FEATURES]

COLUMN_ROLES: dict[str, str] = {
    **{column: "pre_match" for column in PRE_MATCH},
    **{column: "derived_pre_match" for column in DERIVED_PRE_MATCH},
    **{column: "post_match" for column in POST_MATCH + POST_MATCH_NOT_IN_TABLE},
    **{column: "identifier" for column in IDENTIFIERS},
}
FEATURE_ROLES = frozenset({"pre_match", "derived_pre_match"})
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


def assert_valid_features(
    feature_columns: list[str],
    allowed_columns: list[str],
    roles: dict[str, str] = COLUMN_ROLES,
) -> None:
    """Raise unless every feature is allowlisted AND has an explicit pre-match role.

    This is the gate for model inputs. It fails if a feature or allowlist entry:
      - is post-match (the allowlist cannot override the forbidden list);
      - has no role at all, e.g. a newly derived name such as "home_led_at_halftime"
        that nobody has classified (unknown is treated as unsafe);
      - has a non-feature role, e.g. an identifier such as "date";
      - (features only) is not on the allowlist.

    Args:
        roles: Column roles; defaults to COLUMN_ROLES (tests pass their own).
    """
    forbidden = {c for c, role in roles.items() if role == "post_match"}
    for label, columns in (("Features", feature_columns), ("Allowlist", allowed_columns)):
        leaked = sorted(set(columns) & forbidden)
        if leaked:
            raise ValueError(f"{label} contains post-match columns: {leaked}")
        unknown = sorted(c for c in set(columns) if c not in roles)
        if unknown:
            raise ValueError(f"{label} contains columns without a role in COLUMN_ROLES: {unknown}")
        wrong_role = sorted(c for c in set(columns) if roles[c] not in FEATURE_ROLES)
        if wrong_role:
            raise ValueError(f"{label} contains columns that are not pre-match features: {wrong_role}")
    not_allowed = sorted(set(feature_columns) - set(allowed_columns))
    if not_allowed:
        raise ValueError(f"Features not on the allowlist: {not_allowed}")


def result_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count and share of each class, in H, D, A order."""
    counts = df[TARGET].value_counts().reindex(RESULT_CLASSES)
    return pd.DataFrame({"matches": counts, "share": (counts / counts.sum()).round(4)})


# --- Part A: team match log and recent form (Milestone 5) --------------------------

FORM_STATS = {"points": "form_points", "goals_for": "form_goals_for", "goals_against": "form_goals_against"}
FormScope = Literal["carry_over", "season_reset"]


def team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape matches into one row per team per match, from that team's point of view.

    Columns: match_index (the match's index in df), season, date, kickoff_utc, team,
    opponent, is_home (listed first), goals_for, goals_against, points (3/1/0 from the
    90-minute score). Sorted by team, then kickoff time.

    These per-match values are POST-match facts; they may only be used through
    shift(1), i.e. for the team's LATER matches.
    """
    columns = ["season", "date", "kickoff_utc"]
    sides = []
    for side, other, is_home in (("home", "away", True), ("away", "home", False)):
        part = df[columns].copy()
        part["match_index"] = df.index
        part["team"] = df[f"{side}_team"]
        part["opponent"] = df[f"{other}_team"]
        part["is_home"] = is_home
        part["goals_for"] = df[f"{side}_ft"].astype(int)
        part["goals_against"] = df[f"{other}_ft"].astype(int)
        sides.append(part)
    log = pd.concat(sides, ignore_index=True)
    log["points"] = 0
    log.loc[log["goals_for"] > log["goals_against"], "points"] = 3
    log.loc[log["goals_for"] == log["goals_against"], "points"] = 1
    return log.sort_values(["team", "kickoff_utc", "match_index"], kind="stable").reset_index(drop=True)


def add_rolling_form(log: pd.DataFrame, window: int = FORM_WINDOW, scope: FormScope = "carry_over") -> pd.DataFrame:
    """Add form over each team's previous `window` matches, using EARLIER matches only.

    For every row: shift(1) within the team, then a rolling mean over up to `window`
    earlier matches, so a match never sees its own result.

    Args:
        scope: "carry_over" uses the last `window` CL matches whatever the season;
            "season_reset" only uses matches from the same season.

    Adds:
        form_points, form_goals_for, form_goals_against: means over the window
            (missing when there is no earlier match);
        form_n: number of earlier matches actually in the window (0..window);
        form_n_season: how many of those are from the current season.
    """
    out = log.sort_values(["team", "kickoff_utc", "match_index"], kind="stable").copy()
    keys = ["team"] if scope == "carry_over" else ["team", "season"]
    groups = out.groupby(keys, sort=False)
    for stat, name in FORM_STATS.items():
        out[name] = groups[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    out["form_n"] = groups["points"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).count()).fillna(0).astype(int)
    # How many of the window's matches are from this season: compare each earlier
    # match's season (lags 1..window) with the current season.
    same_season = 0
    for lag in range(1, window + 1):
        earlier = groups["season"].shift(lag)
        same_season = same_season + (earlier == out["season"]).astype(int)
    out["form_n_season"] = same_season
    return out


def add_form_features(df: pd.DataFrame, window: int = FORM_WINDOW, scope: FormScope = "carry_over") -> pd.DataFrame:
    """Add home_* and away_* form features (and has_history flags) to each match.

    Teams without earlier matches get missing form values and *_has_history = False;
    nothing is invented.
    """
    form = add_rolling_form(team_match_log(df), window, scope)
    columns = [*FORM_STATS.values(), "form_n", "form_n_season"]
    out = df.copy()
    for side, is_home in (("home", True), ("away", False)):
        part = form[form["is_home"] == is_home].set_index("match_index")[columns]
        part = part.rename(columns=lambda c: f"{side}_{c}")
        out = out.join(part)
        out[f"{side}_has_history"] = out[f"{side}_form_n"] > 0
    return out


# --- Part B: Champions-League-only Elo rating (Milestone 5) -------------------------

ELO_START = 1500.0
ELO_K = 20.0              # update size; fixed in advance, may be tuned on validation data only
ELO_HOME_ADVANTAGE = 60.0  # rating points added to a home side that is really at home
ELO_SEASON_REGRESSION = 1 / 3  # share of the distance to ELO_START removed between seasons


def elo_expected(rating: float, opponent: float, home_advantage: float = 0.0) -> float:
    """Expected score (win 1, draw 0.5, loss 0) of `rating` against `opponent`."""
    return 1.0 / (1.0 + 10 ** ((opponent - (rating + home_advantage)) / 400.0))


def add_elo_features(
    df: pd.DataFrame,
    k: float = ELO_K,
    home_advantage: float = ELO_HOME_ADVANTAGE,
    season_regression: float = ELO_SEASON_REGRESSION,
    start: float = ELO_START,
) -> pd.DataFrame:
    """Add each side's Elo rating BEFORE kickoff, computed sequentially over CL matches.

    For each match in kickoff order: record both pre-match ratings, then update them from
    the 90-minute result. Home advantage applies only when the home side is really at
    home (not neutral_venue, not played_abroad). When a new season starts, every rating
    moves `season_regression` of the way back to `start`. A team's first ever match uses
    `start`; *_elo_n (matches rated before this one) shows how established a rating is.

    Adds: home_elo, away_elo, elo_diff (home - away, without home advantage),
    home_elo_n, away_elo_n.
    """
    order = df.sort_values(["kickoff_utc"], kind="stable").index
    ratings: dict[str, float] = {}
    played: dict[str, int] = {}
    pre = {name: pd.Series(index=df.index, dtype=float) for name in ("home_elo", "away_elo", "home_elo_n", "away_elo_n")}
    season = None

    for index in order:
        row = df.loc[index]
        if row["season"] != season:
            if season is not None:  # summer break: regress every rating toward the mean
                ratings = {team: r - season_regression * (r - start) for team, r in ratings.items()}
            season = row["season"]
        home, away = row["home_team"], row["away_team"]
        r_home, r_away = ratings.get(home, start), ratings.get(away, start)
        pre["home_elo"][index], pre["away_elo"][index] = r_home, r_away
        pre["home_elo_n"][index], pre["away_elo_n"][index] = played.get(home, 0), played.get(away, 0)

        really_home = not (row.get("neutral_venue", False) or row.get("played_abroad", False))
        expected = elo_expected(r_home, r_away, home_advantage if really_home else 0.0)
        actual = 1.0 if row["home_ft"] > row["away_ft"] else 0.5 if row["home_ft"] == row["away_ft"] else 0.0
        change = k * (actual - expected)
        ratings[home], ratings[away] = r_home + change, r_away - change
        played[home], played[away] = played.get(home, 0) + 1, played.get(away, 0) + 1

    out = df.copy()
    for name, values in pre.items():
        out[name] = values.astype(int) if name.endswith("_n") else values
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    return out


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    from src.data.clean import load_clean_matches

    table = add_result_target(load_clean_matches(PROCESSED_DATA_DIR / "matches.csv"))
    print(result_counts(table).to_string())
