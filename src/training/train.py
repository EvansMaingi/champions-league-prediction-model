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

TODO (Milestone 7):
- build_candidate_models() -> dict[str, Pipeline]
    Name -> unfitted Pipeline for each candidate, including the Dummy baseline.
- save_model(model: Pipeline, metadata: dict, path: Path) -> None
    Save with joblib to MODELS_DIR, with metadata such as the training data cutoff.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
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


def build_preprocessor(features: list[str] = ALLOWED_FEATURES) -> ColumnTransformer:
    """Unfitted ColumnTransformer for `features` (see the module docstring).

    Raises if a feature belongs to no branch, so no column is silently dropped.
    """
    unassigned = sorted(set(features) - set(ALLOWED_FEATURES))
    if unassigned:
        raise ValueError(f"Features without a preprocessing branch: {unassigned}")
    def pick(group: list[str]) -> list[str]:
        return [c for c in group if c in features]

    branches = [
        ("form", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), pick(FORM_VALUE_FEATURES)),
        ("first_leg", make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler()),
         pick(FIRST_LEG_FEATURES)),
        ("numeric", StandardScaler(), pick(OTHER_NUMERIC_FEATURES)),
        ("stage", make_pipeline(FunctionTransformer(group_stages, feature_names_out="one-to-one"),
                                OneHotEncoder(handle_unknown="ignore")), pick(["stage"])),
        ("leg_type", OneHotEncoder(handle_unknown="ignore"), pick(["leg_type"])),
    ]
    return ColumnTransformer([branch for branch in branches if branch[2]])


def build_logistic_regression(C: float = 1.0, features: list[str] = ALLOWED_FEATURES) -> Pipeline:
    """Unfitted Pipeline: preprocessor -> multinomial logistic regression (L2, strength 1/C)."""
    return Pipeline([
        ("preprocess", build_preprocessor(features)),
        ("model", LogisticRegression(C=C, max_iter=1000)),
    ])


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
    from src.config import PROCESSED_DATA_DIR
    from src.features.build_features import FEATURE_TABLE_FILE, load_feature_table
    from src.training.evaluate import evaluate_models
    from src.training.split import chronological_split, split_features_target

    # The test set is unpacked as `_` on purpose: it stays sealed until Milestone 8.
    train, validation, _ = chronological_split(load_feature_table(PROCESSED_DATA_DIR / FEATURE_TABLE_FILE))
    X_train, y_train = split_features_target(train)
    X_val, y_val = split_features_target(validation)
    models = train_models({**build_baselines(), "logistic_regression": build_logistic_regression()},
                          X_train, y_train)
    print(f"Fitted on {len(train)} training matches, scored on {len(validation)} "
          f"validation matches (actual H/D/A: "
          f"{', '.join(f'{y_val.eq(c).mean():.3f}' for c in ('H', 'D', 'A'))})")
    print(evaluate_models(models, X_val, y_val).round(4).to_string())
    print("\nSame models on their own training matches (in-sample, optimistic):")
    print(evaluate_models(models, X_train, y_train).round(4).to_string())
