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

TODO (Milestone 7):
- build_preprocessor(numeric_cols: list[str], categorical_cols: list[str])
        -> ColumnTransformer
    Scaling / encoding steps, to be fitted inside a Pipeline.
- build_candidate_models() -> dict[str, Pipeline]
    Name -> unfitted Pipeline for each candidate, including the Dummy baseline.
- save_model(model: Pipeline, metadata: dict, path: Path) -> None
    Save with joblib to MODELS_DIR, with metadata such as the training data cutoff.
"""

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.config import RANDOM_SEED


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
    baselines = train_models(build_baselines(), X_train, y_train)
    print(f"Baselines fitted on {len(train)} training matches, scored on {len(validation)} "
          f"validation matches (actual H/D/A: "
          f"{', '.join(f'{y_val.eq(c).mean():.3f}' for c in ('H', 'D', 'A'))})")
    print(evaluate_models(baselines, X_val, y_val).round(4).to_string())
