"""Train candidate models and save the selected one.

Phase: Milestones 6-7 - Baselines and model training.

Previous stage: split.py produced chronological train / validation sets.
Next stage: evaluate.py scores the models; predict.py loads the saved model.

Every model is wrapped in a scikit-learn Pipeline so preprocessing (scaling, encoding)
is fitted on training data only. Candidates: DummyClassifier (baseline),
LogisticRegression, RandomForestClassifier, and a gradient boosting classifier, all using
RANDOM_SEED. Model selection uses the validation set; the test set is not touched here.

Milestone 6 baselines (they ignore the features entirely):
- uniform: 1/3 for each outcome; the "knows nothing" reference (log loss ln 3).
- prior: the training set's H / D / A shares for every match.
- always_home: 100% home win; scored for accuracy, its log loss shows why hard
  predictions are dangerous.

Milestone 7, part A: build_preprocessor + build_logistic_regression. The preprocessor
has one branch per kind of column (see the *_FEATURES lists below):
- form values: missing only for cold starts -> training median (the *_has_history
  flags, in the "other numeric" branch, tell the model the value was imputed);
- first-leg values: missing when the match is not a second leg -> 0, "not applicable"
  (leg_type tells the model which matches are second legs);
- other numeric (counts, flags, Elo): no missing values;
- stage: mapped to groups that exist in both formats (STAGE_GROUPS), then one-hot;
- leg_type: one-hot.
Numeric branches end with StandardScaler, so every weight is "per standard deviation"
and the L2 penalty treats features alike. Medians, means and standard deviations are
learned in fit(), i.e. from the training matches only.

Milestone 7, part B: build_random_forest and build_hist_gradient_boosting reuse the
preprocessor without scaling (trees split on thresholds); HistGradientBoosting also skips
imputation (it handles NaN itself). build_candidate_models lists every model compared
under walk-forward validation (evaluate.walk_forward_evaluate).

TODO (Milestone 7):
- save_model(model: Pipeline, metadata: dict, path: Path) -> None
    Save with joblib to MODELS_DIR, with metadata such as the training data cutoff.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED
from src.features.build_features import ALLOWED_FEATURES

# --- Column groups for the preprocessor (together exactly ALLOWED_FEATURES) ----------
FORM_VALUE_FEATURES: list[str] = [
    f"{side}_form_{name}" for side in ("home", "away") for name in ("points", "goals_for", "goals_against")
]
FIRST_LEG_FEATURES: list[str] = ["first_leg_goal_diff", "first_leg_away_goals"]
CATEGORICAL_FEATURES: list[str] = ["stage", "leg_type"]
OTHER_NUMERIC_FEATURES: list[str] = [
    c for c in ALLOWED_FEATURES if c not in FORM_VALUE_FEATURES + FIRST_LEG_FEATURES + CATEGORICAL_FEATURES
]

# Raw stage -> a group that exists in training AND in the new 36-team format. Without
# this, the new format's "league" and "playoff" would be unknown categories (all-zero
# one-hot rows) for 160 of the 189 validation matches.
# - group (old format) and league (new): the first phase of the competition;
# - playoff (new) and round_of_16: the first two-legged round where the seeded side
#   hosts the second leg (same grouping as notebooks/02_eda_distributions.ipynb).
STAGE_GROUPS: dict[str, str] = {
    "group": "group_or_league",
    "league": "group_or_league",
    "playoff": "round_of_16_or_playoff",
    "round_of_16": "round_of_16_or_playoff",
    "quarterfinal": "quarterfinal",
    "semifinal": "semifinal",
    "final": "final",
}


def group_stages(stages: pd.DataFrame) -> pd.DataFrame:
    """Map raw stage names to STAGE_GROUPS; raises on a stage that is not listed.

    Raising (instead of letting the encoder ignore it) makes a future format change
    fail loudly rather than silently become an all-zero row.
    """
    grouped = stages.apply(lambda column: column.map(STAGE_GROUPS))
    unknown = sorted(set(stages.to_numpy()[grouped.isna().to_numpy()]))
    if unknown:
        raise ValueError(f"Stages without a group in STAGE_GROUPS: {unknown}")
    return grouped


def to_float(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns as float (pd.NA in Int64 columns -> NaN, booleans -> 0/1)."""
    return X.astype(float)


def build_preprocessor(
    features: list[str] = ALLOWED_FEATURES, scale: bool = True, impute: bool = True,
) -> ColumnTransformer:
    """Unfitted ColumnTransformer for `features` (see the module docstring).

    scale=False drops the StandardScalers (trees split on thresholds, so units do not
    matter). impute=False leaves missing values as NaN for models that handle them
    (HistGradientBoosting). Raises if a feature belongs to no branch, so no column is
    silently dropped.
    """
    unassigned = sorted(set(features) - set(ALLOWED_FEATURES))
    if unassigned:
        raise ValueError(f"Features without a preprocessing branch: {unassigned}")

    def pick(group: list[str]) -> list[str]:
        return [c for c in group if c in features]

    def numeric(imputer: SimpleImputer | None) -> Pipeline:
        first = imputer if impute and imputer is not None else FunctionTransformer(
            to_float, feature_names_out="one-to-one")
        return make_pipeline(first, *([StandardScaler()] if scale else []))

    branches = [
        ("form", numeric(SimpleImputer(strategy="median")), pick(FORM_VALUE_FEATURES)),
        ("first_leg", numeric(SimpleImputer(strategy="constant", fill_value=0)), pick(FIRST_LEG_FEATURES)),
        ("numeric", numeric(None), pick(OTHER_NUMERIC_FEATURES)),
        ("stage", make_pipeline(FunctionTransformer(group_stages, feature_names_out="one-to-one"),
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False)), pick(["stage"])),
        ("leg_type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), pick(["leg_type"])),
    ]
    return ColumnTransformer([branch for branch in branches if branch[2]])


def build_logistic_regression(C: float = 1.0, features: list[str] = ALLOWED_FEATURES) -> Pipeline:
    """Unfitted Pipeline: preprocessor -> multinomial logistic regression (L2, strength 1/C)."""
    return Pipeline([
        ("preprocess", build_preprocessor(features)),
        ("model", LogisticRegression(C=C, max_iter=1000)),
    ])


def build_random_forest(features: list[str] = ALLOWED_FEATURES, **params) -> Pipeline:
    """Unfitted Pipeline: imputation (no scaling) -> Random Forest of 500 trees.

    Other RandomForestClassifier defaults (fully grown trees) unless given in params.
    """
    return Pipeline([
        ("preprocess", build_preprocessor(features, scale=False)),
        ("model", RandomForestClassifier(**{"n_estimators": 500, "random_state": RANDOM_SEED,
                                            "n_jobs": -1, **params})),
    ])


def build_hist_gradient_boosting(features: list[str] = ALLOWED_FEATURES, **params) -> Pipeline:
    """Unfitted Pipeline: no imputation, no scaling -> HistGradientBoostingClassifier.

    Missing values stay NaN: each split learns which side they go to. early_stopping is
    off because it would hold out a RANDOM 10% of the training matches, not later ones.
    """
    return Pipeline([
        ("preprocess", build_preprocessor(features, scale=False, impute=False)),
        ("model", HistGradientBoostingClassifier(**{"random_state": RANDOM_SEED,
                                                    "early_stopping": False, **params})),
    ])


def build_candidate_models() -> dict[str, Pipeline | DummyClassifier]:
    """Name -> unfitted model for the Milestone 7 comparison, prior baseline included."""
    return {
        "prior": build_baselines()["prior"],
        "logistic_regression": build_logistic_regression(),
        "random_forest": build_random_forest(),
        "hist_gradient_boosting": build_hist_gradient_boosting(),
    }


def build_baselines() -> dict[str, DummyClassifier]:
    """Unfitted feature-free baselines (see module docstring)."""
    return {
        "uniform": DummyClassifier(strategy="uniform", random_state=RANDOM_SEED),
        "prior": DummyClassifier(strategy="prior"),
        "always_home": DummyClassifier(strategy="constant", constant="H"),
    }


def train_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit each model on the training data only; returns the same dict."""
    for model in models.values():
        model.fit(X_train, y_train)
    return models


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR, VALIDATION_SEASONS, WALK_FORWARD_SEASONS
    from src.features.build_features import FEATURE_TABLE_FILE, load_feature_table
    from src.training.evaluate import paired_bootstrap, summarise_walk_forward, walk_forward_evaluate
    from src.training.split import chronological_split, walk_forward_folds

    # The test set is unpacked as `_` on purpose: it stays sealed until Milestone 8.
    train, validation, _ = chronological_split(load_feature_table(PROCESSED_DATA_DIR / FEATURE_TABLE_FILE))
    folds = walk_forward_folds(pd.concat([train, validation]), WALK_FORWARD_SEASONS)
    print("Walk-forward folds (expanding window):")
    for season, fit_rows, held_out in folds:
        print(f"  score {season} ({len(held_out)} matches) <- fit on "
              f"{fit_rows['season'].min()} to {fit_rows['season'].max()} ({len(fit_rows)})")
    scores, predictions = walk_forward_evaluate(build_candidate_models(), folds)
    print(f"\nPooled over {predictions['season'].eq(predictions['season']).sum() // scores['model'].nunique()} "
          f"held-out matches (val_*), mean in-sample fold score (train_log_loss):")
    print(summarise_walk_forward(scores, focus_season=VALIDATION_SEASONS[-1]).round(4).to_string())
    print("\nHeld-out log loss per season:")
    held = scores[scores["set"] == "validation"]
    print(held.pivot(index="model", columns="season", values="log_loss").round(4).to_string())
    print("\nPer-match log loss difference, 95% paired bootstrap interval (negative = first model better):")
    for model, reference in [(m, "prior") for m in ("logistic_regression", "random_forest", "hist_gradient_boosting")] + [
            ("random_forest", "logistic_regression"), ("hist_gradient_boosting", "logistic_regression")]:
        b = paired_bootstrap(predictions, model, reference)
        print(f"  {model} - {reference}: {b['mean_diff']:+.4f} [{b['low']:+.4f}, {b['high']:+.4f}]")
