"""Split the feature table into train / validation / test sets by time.

Phase: Milestone 6 - Chronological split and baseline.

Previous stage: build_features.py produced the feature table.
Next stage: train.py fits models on train and compares them on validation.

The split is always chronological (older matches -> train, later -> validation, latest
-> test), never random: a model must not learn from matches played after the ones it
is judged on. We split by whole seasons (src/config.py), so a season's matches never
straddle two sets and there are months between sets. The warm-up season builds form and
Elo history but is in no set.

Features of a validation or test match are still built from every earlier match,
including earlier validation/test matches: those results are known at kickoff. What
must never cross the boundary is anything *learned* (model weights, imputation values).

The test set is sealed: describe_split reports its size and dates but not its results,
and nothing in Milestones 6-7 scores it.
"""

import pandas as pd

from src.config import TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS, WARMUP_SEASONS
from src.features.build_features import (
    ALLOWED_FEATURES, RESULT_CLASSES, TARGET, assert_valid_features,
)

SET_NAMES = ["train", "validation", "test"]


def season_roles(
    warmup: list[str] = WARMUP_SEASONS,
    train: list[str] = TRAIN_SEASONS,
    validation: list[str] = VALIDATION_SEASONS,
    test: list[str] = TEST_SEASONS,
) -> dict[str, str]:
    """Map each season to "warmup", "train", "validation" or "test".

    Raises if a season is given two roles or if any of the three sets is empty.
    """
    roles: dict[str, str] = {}
    for role, seasons in (("warmup", warmup), ("train", train), ("validation", validation), ("test", test)):
        if role != "warmup" and not seasons:
            raise ValueError(f"No seasons assigned to {role}")
        for season in seasons:
            if season in roles:
                raise ValueError(f"Season {season} is in both {roles[season]} and {role}")
            roles[season] = role
    return roles


def chronological_split(
    df: pd.DataFrame, roles: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test), each sorted by date, warm-up rows dropped.

    Raises if a season in `df` has no role (a new season must be placed on purpose,
    not silently dropped or trained on), then checks the result with
    assert_chronological_split.
    """
    roles = season_roles() if roles is None else roles
    unassigned = sorted(set(df["season"]) - set(roles))
    if unassigned:
        raise ValueError(f"Seasons without a split role in src/config.py: {unassigned}")
    role = df["season"].map(roles)
    train, validation, test = (
        df[role == name].sort_values("date", kind="stable") for name in SET_NAMES
    )
    warmup = [season for season, r in roles.items() if r == "warmup"]
    assert_chronological_split(train, validation, test, warmup)
    return train, validation, test


def assert_chronological_split(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
    warmup_seasons: list[str] = WARMUP_SEASONS,
) -> None:
    """Raise unless train < validation < test in time, with no shared or warm-up seasons.

    "<" means the last match date of one set is strictly before the first of the next.
    Calendar dates are used (kickoff_utc can be missing for a few old matches).
    """
    sets = dict(zip(SET_NAMES, (train, validation, test)))
    for name, frame in sets.items():
        if frame.empty:
            raise ValueError(f"The {name} set is empty")
        warm = sorted(set(frame["season"]) & set(warmup_seasons))
        if warm:
            raise ValueError(f"Warm-up seasons {warm} are in the {name} set")
    for (earlier, a), (later, b) in zip(list(sets.items()), list(sets.items())[1:]):
        shared = sorted(set(a["season"]) & set(b["season"]))
        if shared:
            raise ValueError(f"Seasons {shared} are in both {earlier} and {later}")
        if a["date"].max() >= b["date"].min():
            raise ValueError(
                f"{earlier} ends {a['date'].max():%Y-%m-%d}, not before {later} starts "
                f"{b['date'].min():%Y-%m-%d}"
            )


def split_features_target(
    df: pd.DataFrame, feature_cols: list[str] = ALLOWED_FEATURES,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate X (model inputs) from y (`result` as "H" / "D" / "A" strings).

    The features pass the Milestone 4/5 gate first, so a post-match or unclassified
    column can never reach a model. y is plain strings because scikit-learn sorts the
    labels into classes_ = ["A", "D", "H"] either way.
    """
    assert_valid_features(feature_cols, ALLOWED_FEATURES)
    return df[feature_cols].copy(), df[TARGET].astype(str)


def describe_split(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per set: seasons, match count, date range and result shares.

    The test set's result shares are left blank: it stays sealed until Milestone 8.
    """
    rows = []
    for name, frame in zip(SET_NAMES, (train, validation, test)):
        first, last = frame["season"].min(), frame["season"].max()
        row = {
            "set": name,
            "seasons": first if first == last else f"{first} to {last}",
            "matches": len(frame),
            "first": frame["date"].min().date(),
            "last": frame["date"].max().date(),
        }
        shares = frame[TARGET].value_counts(normalize=True)
        for cls in RESULT_CLASSES:
            row[f"{cls}%"] = None if name == "test" else round(100 * shares.get(cls, 0.0), 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("set")


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    from src.features.build_features import FEATURE_TABLE_FILE, load_feature_table

    sets = chronological_split(load_feature_table(PROCESSED_DATA_DIR / FEATURE_TABLE_FILE))
    print(describe_split(*sets).to_string())
    print(f"Training data cutoff: {sets[0]['date'].max():%Y-%m-%d}")
