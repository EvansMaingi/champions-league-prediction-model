"""Train candidate models and save the selected one.

Phase: Milestone 7 - Model training.

Previous stage: split.py produced chronological train / validation sets.
Next stage: evaluate.py scores the models; predict.py loads the saved model.

Every model is wrapped in a scikit-learn Pipeline so preprocessing (scaling, encoding)
is fitted on training data only. Candidates: DummyClassifier (baseline),
LogisticRegression, RandomForestClassifier, and a gradient boosting classifier, all using
RANDOM_SEED. Model selection uses the validation set; the test set is not touched here.

TODO (planned functions):
- build_preprocessor(numeric_cols: list[str], categorical_cols: list[str])
        -> ColumnTransformer
    Scaling / encoding steps, to be fitted inside a Pipeline.
- build_candidate_models() -> dict[str, Pipeline]
    Name -> unfitted Pipeline for each candidate, including the Dummy baseline.
- train_models(models: dict[str, Pipeline], X_train, y_train) -> dict[str, Pipeline]
    Fit each pipeline on training data.
- save_model(model: Pipeline, metadata: dict, path: Path) -> None
    Save with joblib to MODELS_DIR, with metadata such as the training data cutoff.
"""
