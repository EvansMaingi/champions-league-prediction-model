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

Walk-forward (Milestone 7): walk_forward_evaluate refits a fresh copy of every model on
each fold (split.walk_forward_folds) and scores both the fold's training matches
(in-sample, to show overfitting) and its held-out season. summarise_walk_forward pools
the held-out seasons, weighting each by its number of matches; paired_bootstrap gives
an interval for the per-match log loss difference between two models.

TODO (Milestone 8):
- plot_confusion_matrix(model, X, y) -> matplotlib.figure.Figure
    Actual vs predicted H / D / A counts.
- plot_calibration(model, X, y) -> matplotlib.figure.Figure
    Reliability curve per outcome class (H / D / A), especially D (draw-rate shift).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss

from src.config import RANDOM_SEED
from src.features.build_features import ALLOWED_FEATURES, RESULT_CLASSES
from src.training.split import split_features_target


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
    """Log loss, Brier score, accuracy, the mean predicted probability of each class, and
    share_over_90 (share of matches where one outcome gets more than 0.9).

    Accuracy takes the most likely class per match; ties go to the first column (H).
    Comparing mean_p_D with the actual draw rate shows draws being over/under-predicted.
    Log loss clips probabilities away from 0, so a hard "0% chance" that happens costs
    about 36 nats for that match instead of infinity. share_over_90 flags severe
    overconfidence (selection rule: more than 2% excludes a model).
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
    metrics["share_over_90"] = float((proba.max(axis=1) > 0.9).mean())
    return metrics


def evaluate_models(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """One row per fitted model on the same matches, sorted by log loss (best first)."""
    rows = {name: evaluate_probabilities(y, predict_proba_frame(model, X)) for name, model in models.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("log_loss")


def per_match_log_loss(y_true: pd.Series, proba: pd.DataFrame) -> np.ndarray:
    """-ln(probability given to the actual result), per match.

    Clipped like scikit-learn's log_loss (at float eps), so its mean equals log_loss.
    """
    p = proba.to_numpy()[np.arange(len(y_true)), proba.columns.get_indexer(y_true)]
    eps = np.finfo(float).eps
    return -np.log(np.clip(p, eps, 1 - eps))


def walk_forward_evaluate(
    models: dict, folds: list, extra_features: list[str] = (), sample_weight=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit a fresh clone of every model on each fold and score it.

    folds: output of split.walk_forward_folds, [(season, fit_rows, score_rows), ...].
    extra_features: columns passed to the models besides ALLOWED_FEATURES (an
    experiment, e.g. team names). sample_weight: optional function fit_rows -> weights,
    passed to the final model step when fitting (in-sample scores stay unweighted).
    Returns (scores, predictions):
    - scores: one row per (model, season, set), set "train" (in-sample) or "validation",
      with n and the evaluate_probabilities metrics;
    - predictions: one row per (model, held-out match), indexed like the feature table,
      with model, season, result and the H / D / A probabilities.
    """
    features = [*ALLOWED_FEATURES, *extra_features]
    score_rows, prediction_frames = [], []
    for season, fit_rows, held_out in folds:
        X_fit, y_fit = split_features_target(fit_rows, features, allowed=features)
        X_val, y_val = split_features_target(held_out, features, allowed=features)
        fit_params = {} if sample_weight is None else {"model__sample_weight": sample_weight(fit_rows)}
        for name, model in models.items():
            # clone: an unfitted copy, so nothing is carried from one fold to the next
            fitted = clone(model).fit(X_fit, y_fit, **fit_params)
            for set_name, X, y in (("train", X_fit, y_fit), ("validation", X_val, y_val)):
                proba = predict_proba_frame(fitted, X)
                score_rows.append({"model": name, "season": season, "set": set_name, "n": len(y),
                                   **evaluate_probabilities(y, proba)})
            prediction_frames.append(proba.assign(model=name, season=season, result=y_val))
    return pd.DataFrame(score_rows), pd.concat(prediction_frames)


def summarise_walk_forward(scores: pd.DataFrame, focus_season: str | None = None) -> pd.DataFrame:
    """One row per model, best pooled validation log loss first.

    val_*: held-out seasons pooled, each weighted by its number of matches (the same as
    scoring all held-out matches together). train_log_loss: mean in-sample log loss
    over the folds; gap = val_log_loss - train_log_loss (large gap = overfitting).
    focus_season adds that fold's held-out log loss as its own column.
    """
    metrics = ["log_loss", "brier", "accuracy", "mean_p_D", "share_over_90"]
    val = scores[scores["set"] == "validation"]
    weighted = val[metrics].mul(val["n"], axis=0).groupby(val["model"]).sum()
    summary = weighted.div(val.groupby("model")["n"].sum(), axis=0).add_prefix("val_")
    summary["train_log_loss"] = scores[scores["set"] == "train"].groupby("model")["log_loss"].mean()
    summary["gap"] = summary["val_log_loss"] - summary["train_log_loss"]
    if focus_season is not None:
        focus = val[val["season"] == focus_season].set_index("model")["log_loss"]
        summary[f"log_loss_{focus_season}"] = focus
    return summary.sort_values("val_log_loss")


def paired_bootstrap(
    predictions: pd.DataFrame, model: str, reference: str, n_boot: int = 5000, seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Mean per-match log loss of `model` minus `reference` on the same held-out matches,
    with a 95% percentile interval from resampling matches (negative = model better).

    Matches are resampled independently, ignoring that matches of one season share
    conditions, so the interval is, if anything, a little too narrow.
    """
    losses = {}
    for name in (model, reference):
        rows = predictions[predictions["model"] == name].sort_index()
        losses[name] = per_match_log_loss(rows["result"], rows[RESULT_CLASSES])
    diff = losses[model] - losses[reference]
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return {"mean_diff": float(diff.mean()), "low": float(np.quantile(boots, 0.025)),
            "high": float(np.quantile(boots, 0.975)), "n": len(diff)}
