"""Evaluate predicted probabilities and calibration on held-out matches.

Phase: Milestones 6-8 - Baselines, evaluation and probability calibration.

Previous stage: train.py produced fitted models / pipelines.
Next stage: the chosen (possibly calibrated) model is used by predict.py.

Probabilities are judged with log loss and multi-class Brier score; class predictions
with accuracy. Every model is compared against the DummyClassifier baselines on the
same matches. The test set is evaluated once, at the very end of Milestone 8.

Column order: scikit-learn sorts labels, so predict_proba columns follow
model.classes_ = ["A", "D", "H"]. predict_proba_frame renames them by label and returns
H, D, A, so a column can never be read as the wrong outcome.

TODO (Milestone 8):
- plot_confusion_matrix(model, X, y) -> matplotlib.figure.Figure
    Actual vs predicted H / D / A counts.
- plot_calibration(model, X, y) -> matplotlib.figure.Figure
    Reliability curve per outcome class (H / D / A), especially D (draw-rate shift).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.features.build_features import RESULT_CLASSES


def predict_proba_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    """Predicted probabilities as a DataFrame with columns H, D, A (in that order).

    Columns are matched by the labels in model.classes_, never by position. Raises if
    the model was trained on a different set of labels.
    """
    classes = [str(c) for c in model.classes_]
    if sorted(classes) != sorted(RESULT_CLASSES):
        raise ValueError(f"Model classes {classes} are not {RESULT_CLASSES}")
    proba = pd.DataFrame(model.predict_proba(X), columns=classes, index=X.index)
    return proba[RESULT_CLASSES]


def multiclass_brier_score(y_true: pd.Series, proba: pd.DataFrame) -> float:
    """Mean over matches of the summed squared error against the one-hot outcome.

    Range 0 (certain and right) to 2 (certain and wrong); uniform 1/3 scores 2/3.
    """
    onehot = np.column_stack([(y_true == cls).to_numpy() for cls in proba.columns]).astype(float)
    return float(((proba.to_numpy() - onehot) ** 2).sum(axis=1).mean())


def evaluate_probabilities(y_true: pd.Series, proba: pd.DataFrame) -> dict[str, float]:
    """Log loss, Brier score, accuracy, and the mean predicted probability of each class.

    Accuracy takes the most likely class per match; ties go to the first column (H).
    Comparing mean_p_D with the actual draw rate shows draws being over/under-predicted.
    Log loss clips probabilities away from 0, so a hard "0% chance" that happens costs
    about 36 nats for that match instead of infinity.
    """
    predicted = proba.idxmax(axis=1)
    # scikit-learn assumes probability columns are in sorted label order (A, D, H) and
    # does NOT reorder them to match `labels`, so sort the columns first.
    sorted_labels = sorted(proba.columns)
    metrics = {
        "log_loss": log_loss(y_true, proba[sorted_labels].to_numpy(), labels=sorted_labels),
        "brier": multiclass_brier_score(y_true, proba),
        "accuracy": float((predicted == y_true).mean()),
    }
    for cls in proba.columns:
        metrics[f"mean_p_{cls}"] = float(proba[cls].mean())
    return metrics


def evaluate_models(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """One row per fitted model on the same matches, sorted by log loss (best first)."""
    rows = {name: evaluate_probabilities(y, predict_proba_frame(model, X)) for name, model in models.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("log_loss")
