"""Split the feature table into train / validation / test sets by time.

Phase: Milestone 6 - Chronological split and baseline.

Previous stage: build_features.py produced the feature table.
Next stage: train.py fits models on train and compares them on validation.

The split is always chronological (older matches -> train, later -> validation, latest
-> test), never random: a model must not learn from matches played after the ones it
is judged on. The cutoff dates are recorded so we can report the training data cutoff.

TODO (planned functions):
- chronological_split(df: pd.DataFrame, val_start: str, test_start: str)
        -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    Return (train, validation, test) split at the given dates.
- split_features_target(df: pd.DataFrame, feature_cols: list[str])
        -> tuple[pd.DataFrame, pd.Series]
    Separate X (features) from y (`result`).
- describe_split(train, validation, test) -> pd.DataFrame
    Summarise date range, row count and class balance of each set.
"""
