"""Check that the cleaned match table is trustworthy before it is used.

Phase: Milestones 2-3 - Data validation.

Previous stage: clean.py produced the clean match table.
Next stage: build_features.py builds features from a table that passed validation.

Validation fails loudly rather than silently fixing problems: missing required columns,
null kickoff dates, a team playing itself, duplicate fixtures, or scores that are
negative or non-integer.

TODO (planned functions):
- check_required_columns(df: pd.DataFrame, required: list[str]) -> None
    Raise ValueError naming any missing column.
- check_no_duplicate_matches(df: pd.DataFrame) -> None
    Raise if the same (date, home_team, away_team) appears more than once.
- check_scores_valid(df: pd.DataFrame) -> None
    Raise if goals are negative, non-integer, or missing for played matches.
- validate_matches(df: pd.DataFrame) -> pd.DataFrame
    Run every check and return df unchanged if all pass.
"""
