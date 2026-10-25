"""Check that the cleaned match table is trustworthy before it is used.

Phase: Milestone 3 - Data validation.

Previous stage: clean.py produced the clean match table.
Next stage: build_features.py builds features from a table that passed validation.

Validation fails loudly rather than silently fixing problems: each check raises a
ValueError that names the offending rows, and returns nothing when all is well.
validate_matches runs every match-table check; validate_team_table checks the
reviewed team-name table.
"""

import pandas as pd

REQUIRED_COLUMNS = [
    "season", "date", "kickoff_cet", "kickoff_utc", "stage", "home_team", "away_team",
    "home_country", "away_country", "home_ft", "away_ft", "home_ht", "away_ht",
    "ht_inferred", "neutral_venue", "played_abroad", "behind_closed_doors", "tie_id", "leg",
]


def check_required_columns(df: pd.DataFrame, required: list[str] = REQUIRED_COLUMNS) -> None:
    """Raise ValueError naming any missing column."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def check_no_duplicate_matches(df: pd.DataFrame) -> None:
    """Raise if the same (date, home_team, away_team) appears more than once."""
    duplicated = df.duplicated(subset=["date", "home_team", "away_team"], keep=False)
    if duplicated.any():
        raise ValueError(f"Duplicate matches: rows {df.index[duplicated].tolist()}")


def check_teams_mapped(df: pd.DataFrame, mapping: dict[str, str]) -> None:
    """Raise if any home or away team name is not a key of the team-name mapping.

    An unmapped name would otherwise become a "new club" with no history, which is
    exactly the silent error the reviewed mapping exists to prevent.
    """
    names = set(df["home_team"]) | set(df["away_team"])
    unmapped = sorted(names - set(mapping))
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} team name(s) missing from data/reference/team_names.csv: {unmapped}"
        )


def validate_team_table(table: pd.DataFrame) -> None:
    """Raise if the team-name table is incomplete or ambiguous.

    Every row needs a standard name and a country; each club (standard name) has
    exactly one country and one Wikipedia article, and each article one standard name.
    """
    problems = []
    for column in ("standard_name", "country"):
        empty = table[table[column].fillna("").str.strip() == ""]
        if not empty.empty:
            problems.append(f"rows without {column}: {empty['source_name'].tolist()}")
    for column, other in (("standard_name", "country"), ("standard_name", "wikipedia_article"),
                          ("wikipedia_article", "standard_name")):
        counts = table[table[column] != ""].groupby(column)[other].nunique()
        if (counts > 1).any():
            problems.append(f"{column} with more than one {other}: {counts[counts > 1].index.tolist()}")
    if problems:
        raise ValueError("; ".join(problems))


def check_scores_consistent(df: pd.DataFrame) -> None:
    """Raise if a match's scores contradict each other.

    Rules: goals are never negative; half-time <= 90 minutes <= after extra time for
    each side. (Penalty shootouts depend on the whole tie: see check_shootouts.)
    """
    problems = []
    for side in ("home", "away"):
        ht, ft, aet = df[f"{side}_ht"], df[f"{side}_ft"], df[f"{side}_aet"]
        for name, bad in {
            f"negative {side} goals": (ft < 0) | (ht < 0) | (aet < 0),
            f"{side} half-time > 90 minutes": ht > ft,
            f"{side} 90 minutes > extra time": ft > aet,
        }.items():
            bad = bad.fillna(False)
            if bad.any():
                problems.append(f"{name}: rows {df.index[bad].tolist()}")
    if problems:
        raise ValueError("; ".join(problems))


def final_goals(row: pd.Series, side: str) -> int:
    """Goals after extra time if it was played, else after 90 minutes."""
    return row[f"{side}_aet"] if pd.notna(row[f"{side}_aet"]) else row[f"{side}_ft"]


def check_shootouts(df: pd.DataFrame) -> None:
    """Raise if a penalty shootout did not follow a level tie (needs tie_id and leg).

    Single match: the score after extra time must be level.
    Two legs: the shootout is in leg 2 and the aggregate over both legs is level.
    """
    problems = []
    for index, row in df[df["home_pens"].notna()].iterrows():
        if row["leg"] == "single":
            level = final_goals(row, "home") == final_goals(row, "away")
        elif row["leg"] == "2":
            first = df[(df["tie_id"] == row["tie_id"]) & (df["leg"] == "1")].iloc[0]
            # The leg-2 home team was the leg-1 away team.
            level = (final_goals(row, "home") + final_goals(first, "away")
                     == final_goals(row, "away") + final_goals(first, "home"))
        else:
            level = False
        if not level:
            problems.append(f"row {index} ({row['tie_id']}, leg {row['leg']})")
    if problems:
        raise ValueError(f"Penalty shootout after a tie that was not level: {problems}")


def check_chronological(df: pd.DataFrame) -> None:
    """Raise if the table is not sorted by date (later stages rely on time order)."""
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Match table is not sorted chronologically by date")


def validate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Run every match-table check; return df unchanged if all pass."""
    check_required_columns(df)
    check_no_duplicate_matches(df)
    check_chronological(df)
    check_scores_consistent(df)
    check_shootouts(df)
    return df
