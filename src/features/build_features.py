"""Create the target and pre-match features for every match.

Phase: Milestone 4 - Target definition, and Milestone 5 - Feature engineering.

Previous stage: validate.py confirmed the clean match table.
Next stage: split.py divides this feature table chronologically.

Leakage rules (most important part of the project):
- Every feature must be known BEFORE kickoff. Final scores, shots and possession of
  the match itself are never inputs. The target `result` is never a feature.
- Rolling features are computed on chronologically sorted data from earlier matches
  only, and shifted by one so a match never influences its own features.
- Teams with fewer than FORM_WINDOW previous matches are handled explicitly
  (e.g. partial windows plus a "matches played" count), not silently filled.

TODO (planned functions):
- add_result_target(df: pd.DataFrame) -> pd.DataFrame
    Add `result` = "H" / "D" / "A" from home and away goals.
- to_team_long_format(df: pd.DataFrame) -> pd.DataFrame
    One row per (match, team) so each team's history can be rolled.
- add_rolling_form(long_df: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame
    Shifted rolling means (points, goals for/against) from previous matches only.
- merge_home_away_features(df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame
    Attach home-team and away-team features back onto each match row.
- build_feature_table(df: pd.DataFrame) -> pd.DataFrame
    Run all steps and save the feature table to PROCESSED_DATA_DIR.
"""
