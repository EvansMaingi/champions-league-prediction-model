"""Tests for src/training/split.py, train.py baselines and evaluate.py metrics (Milestone 6)."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from src.config import PROCESSED_DATA_DIR
from src.features.build_features import ALLOWED_FEATURES, FEATURE_TABLE_FILE, load_feature_table
from src.training.evaluate import (
    evaluate_probabilities, multiclass_brier_score, predict_proba_frame,
)
from src.training.split import (
    assert_chronological_split, chronological_split, describe_split, season_roles,
    split_features_target,
)
from src.training.train import build_baselines, train_models

ROLES = season_roles(warmup=["2010-11"], train=["2011-12", "2012-13"], validation=["2013-14"], test=["2014-15"])


def season_table(seasons=("2010-11", "2011-12", "2012-13", "2013-14", "2014-15"), per_season=3) -> pd.DataFrame:
    """Small fake feature table: `per_season` matches each autumn, results cycling H, D, A."""
    rows = []
    for season in seasons:
        start = pd.Timestamp(f"{season[:4]}-09-15")
        for i in range(per_season):
            rows.append({"season": season, "date": start + pd.Timedelta(days=30 * i), "result": "HDA"[i % 3]})
    df = pd.DataFrame(rows)
    for column in ALLOWED_FEATURES:
        df[column] = 0.0
    return df


# --- season roles and the split ----------------------------------------------------

def test_split_puts_each_season_in_one_set_and_drops_warmup():
    df = season_table()
    train, validation, test = chronological_split(df, ROLES)
    assert sorted(train["season"].unique()) == ["2011-12", "2012-13"]
    assert list(validation["season"].unique()) == ["2013-14"]
    assert list(test["season"].unique()) == ["2014-15"]
    assert len(train) + len(validation) + len(test) == len(df) - 3  # only the warm-up is lost
    combined = pd.concat([train, validation, test])
    assert not combined.index.duplicated().any()  # no match in two sets


def test_every_train_match_is_before_validation_and_validation_before_test():
    train, validation, test = chronological_split(season_table(), ROLES)
    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()


def test_split_is_independent_of_row_order():
    df = season_table()
    shuffled = df.sample(frac=1, random_state=0)
    for a, b in zip(chronological_split(df, ROLES), chronological_split(shuffled, ROLES)):
        pd.testing.assert_frame_equal(a, b)


def test_season_without_a_role_raises():
    df = season_table(seasons=("2011-12", "2012-13", "2013-14", "2014-15", "2015-16"))
    with pytest.raises(ValueError, match="without a split role.*2015-16"):
        chronological_split(df, ROLES)


def test_season_with_two_roles_raises():
    with pytest.raises(ValueError, match="2013-14 is in both train and validation"):
        season_roles(warmup=[], train=["2013-14"], validation=["2013-14"], test=["2014-15"])


def test_empty_role_raises():
    with pytest.raises(ValueError, match="No seasons assigned to test"):
        season_roles(warmup=[], train=["2012-13"], validation=["2013-14"], test=[])


def test_overlapping_dates_raise():
    # A deliberately broken split: a validation season placed BEFORE a training season.
    roles = season_roles(warmup=[], train=["2011-12", "2013-14"], validation=["2012-13"], test=["2014-15"])
    with pytest.raises(ValueError, match="train ends 2013-11-14, not before validation starts 2012-09-15"):
        chronological_split(season_table(seasons=("2011-12", "2012-13", "2013-14", "2014-15")), roles)


def test_warmup_rows_in_a_set_raise():
    train, validation, test = chronological_split(season_table(), ROLES)
    warmup = season_table(seasons=("2010-11",))
    with pytest.raises(ValueError, match=r"Warm-up seasons \['2010-11'\] are in the train set"):
        assert_chronological_split(pd.concat([warmup, train]), validation, test, ["2010-11"])


def test_shared_season_raises_even_with_ordered_dates():
    train, validation, test = chronological_split(season_table(), ROLES)
    with pytest.raises(ValueError, match=r"Seasons \['2013-14'\] are in both validation and test"):
        assert_chronological_split(train, validation.iloc[:2], pd.concat([validation.iloc[2:], test]), [])


def test_describe_split_keeps_test_results_sealed():
    summary = describe_split(*chronological_split(season_table(), ROLES))
    assert summary.loc["train", "matches"] == 6
    assert summary.loc["validation", ["H%", "D%", "A%"]].tolist() == [33.3, 33.3, 33.3]
    assert summary.loc["test", ["H%", "D%", "A%"]].isna().all()


# --- X / y ---------------------------------------------------------------------------

def test_split_features_target_returns_only_allowed_features():
    X, y = split_features_target(season_table())
    assert list(X.columns) == ALLOWED_FEATURES
    assert "result" not in X.columns
    assert y.tolist()[:3] == ["H", "D", "A"]


def test_split_features_target_rejects_the_target_as_a_feature():
    with pytest.raises(ValueError, match="post-match"):
        split_features_target(season_table(), ALLOWED_FEATURES + ["result"])


# --- metrics -----------------------------------------------------------------------

class FakeModel:
    """Pretends to be a fitted classifier with scikit-learn's sorted label order."""
    classes_ = np.array(["A", "D", "H"])

    def predict_proba(self, X):
        return np.tile([0.1, 0.2, 0.7], (len(X), 1))  # A, D, H


def test_predict_proba_frame_maps_columns_by_label_not_position():
    proba = predict_proba_frame(FakeModel(), pd.DataFrame({"x": [0, 1]}))
    assert list(proba.columns) == ["H", "D", "A"]
    assert proba.iloc[0].tolist() == [0.7, 0.2, 0.1]


def test_predict_proba_frame_rejects_wrong_labels():
    model = FakeModel()
    model.classes_ = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="are not"):
        predict_proba_frame(model, pd.DataFrame({"x": [0]}))


def test_brier_score_hand_examples():
    y = pd.Series(["H", "D"])
    certain_right = pd.DataFrame({"H": [1.0, 0.0], "D": [0.0, 1.0], "A": [0.0, 0.0]})
    certain_wrong = pd.DataFrame({"H": [0.0, 0.0], "D": [0.0, 0.0], "A": [1.0, 1.0]})
    uniform = pd.DataFrame({c: [1 / 3, 1 / 3] for c in "HDA"})
    assert multiclass_brier_score(y, certain_right) == 0.0
    assert multiclass_brier_score(y, certain_wrong) == 2.0
    assert multiclass_brier_score(y, uniform) == pytest.approx(2 / 3)
    # Match 1: H with (0.5, 0.3, 0.2) -> 0.25 + 0.09 + 0.04 = 0.38
    assert multiclass_brier_score(pd.Series(["H"]), pd.DataFrame({"H": [0.5], "D": [0.3], "A": [0.2]})) == pytest.approx(0.38)


def test_brier_score_matches_scikit_learn():
    rng = np.random.default_rng(0)
    raw = rng.random((50, 3))
    proba = pd.DataFrame(raw / raw.sum(axis=1, keepdims=True), columns=["H", "D", "A"])
    y = pd.Series(rng.choice(["H", "D", "A"], 50))
    # scikit-learn's multiclass Brier uses the same definition (sum over classes), but it
    # expects columns in sorted label order A, D, H.
    expected = brier_score_loss(y, proba[["A", "D", "H"]].to_numpy(), labels=["A", "D", "H"])
    assert multiclass_brier_score(y, proba) == pytest.approx(expected)


def test_log_loss_uses_the_right_column_for_each_label():
    # p(H)=0.6, p(D)=0.3, p(A)=0.1; results H then D -> -(ln 0.6 + ln 0.3) / 2.
    # Reading the H, D, A columns as sklearn's A, D, H would score H with 0.1 instead.
    proba = pd.DataFrame({"H": [0.6, 0.6], "D": [0.3, 0.3], "A": [0.1, 0.1]})
    metrics = evaluate_probabilities(pd.Series(["H", "D"]), proba)
    assert metrics["log_loss"] == pytest.approx(-(math.log(0.6) + math.log(0.3)) / 2)


def test_uniform_log_loss_is_ln3_and_accuracy_tie_goes_to_home():
    y = pd.Series(["H", "D", "A", "A"])
    metrics = evaluate_probabilities(y, pd.DataFrame({c: [1 / 3] * 4 for c in "HDA"}))
    assert metrics["log_loss"] == pytest.approx(math.log(3))
    assert metrics["accuracy"] == 0.25  # every tie -> "H"
    assert metrics["mean_p_D"] == pytest.approx(1 / 3)


# --- baselines ---------------------------------------------------------------------

def test_baselines_predict_what_they_should():
    X = pd.DataFrame({"x": range(5)})
    y = pd.Series(["H", "H", "D", "A", "H"])
    models = train_models(build_baselines(), X, y)
    first = {name: predict_proba_frame(m, X).iloc[0].tolist() for name, m in models.items()}
    assert first["uniform"] == pytest.approx([1 / 3] * 3)
    assert first["prior"] == pytest.approx([0.6, 0.2, 0.2])  # training shares H, D, A
    assert first["always_home"] == [1.0, 0.0, 0.0]


# --- the real feature table -------------------------------------------------------

def test_real_split_counts():
    path = PROCESSED_DATA_DIR / FEATURE_TABLE_FILE
    if not path.exists():
        pytest.skip("features.csv not built")
    train, validation, test = chronological_split(load_feature_table(path))
    assert (len(train), len(validation), len(test)) == (1494, 189, 189)
    assert train["season"].min() == "2012-13" and train["season"].max() == "2023-24"
