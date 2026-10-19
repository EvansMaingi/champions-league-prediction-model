"""Turn raw match files into one tidy, consistent match table.

Phase: Milestones 2-3 - Data cleaning (informed by exploratory data analysis).

Previous stage: collect.py saved raw files in data/raw/.
Next stage: validate.py checks this table; build_features.py then uses it.

Cleaning includes standardising column names, parsing kickoff dates, normalising team
names (one spelling per club), and removing exact duplicates. The output is saved to
data/processed/; raw files are never modified.

TODO (planned functions):
- load_raw(path: Path) -> pd.DataFrame
    Read one raw file into a DataFrame.
- standardise_columns(df: pd.DataFrame) -> pd.DataFrame
    Rename columns to the project schema (date, season, home_team, away_team, ...).
- normalise_team_names(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame
    Map spelling variants to one canonical team name.
- parse_dates(df: pd.DataFrame) -> pd.DataFrame
    Convert date strings to datetime and sort chronologically.
- clean_matches(raw_paths: list[Path]) -> pd.DataFrame
    Run all steps, save to PROCESSED_DATA_DIR, and return the clean table.
"""
