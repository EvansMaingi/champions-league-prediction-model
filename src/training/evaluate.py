"""Evaluate predicted probabilities and calibration on held-out matches.

Phase: Milestone 8 - Evaluation and probability calibration.

Previous stage: train.py produced fitted pipelines.
Next stage: the chosen (possibly calibrated) model is used by predict.py.

Probabilities are judged with log loss and multi-class Brier score; class predictions
with accuracy, precision, recall, macro-F1 and a confusion matrix. Every model is
compared against the DummyClassifier baseline on the same matches.
Calibration curves show whether "40% home win" really happens about 40% of the time.
The test set is evaluated once, at the very end.

TODO (planned functions):
- multiclass_brier_score(y_true: pd.Series, proba: np.ndarray, classes: list[str]) -> float
    Mean squared error between one-hot outcomes and predicted probabilities.
- evaluate_model(model: Pipeline, X, y) -> dict[str, float]
    Log loss, Brier score, accuracy and macro-F1 for one model.
- plot_confusion_matrix(model: Pipeline, X, y) -> matplotlib.figure.Figure
    Actual vs predicted H / D / A counts.
- compare_models(models: dict[str, Pipeline], X, y) -> pd.DataFrame
    One row per model, including the baseline, sorted by log loss.
- plot_calibration(model: Pipeline, X, y) -> matplotlib.figure.Figure
    Reliability curve per outcome class (H / D / A).
"""
