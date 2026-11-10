"""Tests for the Milestone 7 preprocessing and model pipelines in src/training/train.py.

Fitted on the real training seasons; validation rows are used only for their pre-match
inputs (stage, missing values), never their results. The test season is not loaded.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DATA_DIR
from src.features.build_features import (
    ALLOWED_FEATURES, FEATURE_TABLE_FILE, IDENTITY_FEATURES, RESULT_CLASSES, load_feature_table,
)
from src.training.evaluate import predict_proba_frame
from src.training.split import chronological_split, split_features_target
from src.training.train import (
    CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, FIRST_LEG_FEATURES, FLAG_FEATURES, FORM_VALUE_FEATURES,
    STAGE_GROUPS, load_model, save_model, season_age_weights, build_candidate_models, build_hist_gradient_boosting, build_logistic_regression,
    build_preprocessor, build_random_forest, group_stages,
)


@pytest.fixture(scope="module")
def train_and_validation_inputs():
    """(X_train, y_train, X_val) from the real feature table."""
    train, validation, _ = chronological_split(load_feature_table(PROCESSED_DATA_DIR / FEATURE_TABLE_FILE))
    X_train, y_train = split_features_target(train)
    X_val, _ = split_features_target(validation)
    return X_train, y_train, X_val


@pytest.fixture(scope="module")
def fitted_lr(train_and_validation_inputs):
    X_train, y_train, _ = train_and_validation_inputs
    return build_logistic_regression().fit(X_train, y_train)


# --- column groups and stage mapping ------------------------------------------------

def test_column_groups_cover_allowed_features_exactly_once():
    groups = FORM_VALUE_FEATURES + FIRST_LEG_FEATURES + CONTINUOUS_FEATURES + FLAG_FEATURES + CATEGORICAL_FEATURES
    assert len(groups) == len(set(groups))
    assert set(groups) == set(ALLOWED_FEATURES)


def test_group_stages_maps_both_formats_to_shared_groups():
    stages = pd.DataFrame({"stage": ["group", "league", "playoff", "round_of_16", "final"]})
    assert group_stages(stages)["stage"].tolist() == [
        "group_or_league", "group_or_league", "round_of_16_or_playoff", "round_of_16_or_playoff", "final"]


def test_group_stages_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="super_league"):
        group_stages(pd.DataFrame({"stage": ["league", "super_league"]}))


def test_every_real_stage_has_a_group(train_and_validation_inputs):
    X_train, _, X_val = train_and_validation_inputs
    assert set(X_train["stage"]) | set(X_val["stage"]) <= set(STAGE_GROUPS)


def test_preprocessor_rejects_a_feature_without_a_branch():
    with pytest.raises(ValueError, match="home_country"):
        build_preprocessor([*ALLOWED_FEATURES, "home_country"])


# --- the fitted preprocessor ---------------------------------------------------------

def test_preprocessor_keeps_every_feature_and_leaves_no_missing_values(fitted_lr, train_and_validation_inputs):
    _, _, X_val = train_and_validation_inputs
    preprocess = fitted_lr.named_steps["preprocess"]
    out = preprocess.transform(X_val)
    numeric = len(FORM_VALUE_FEATURES) + len(FIRST_LEG_FEATURES) + len(CONTINUOUS_FEATURES) + len(FLAG_FEATURES)
    stage_groups = len(set(STAGE_GROUPS.values()))
    leg_types = 4  # "1", "2", "single", "none"
    assert out.shape == (len(X_val), numeric + stage_groups + leg_types)
    assert not np.isnan(out).any()


def test_imputation_values_are_learned_from_training_data(fitted_lr, train_and_validation_inputs):
    X_train, _, _ = train_and_validation_inputs
    transformers = fitted_lr.named_steps["preprocess"].named_transformers_
    form_imputer = transformers["form"].named_steps["simpleimputer"]
    np.testing.assert_allclose(form_imputer.statistics_, X_train[FORM_VALUE_FEATURES].median().to_numpy())
    # First-leg features: "not applicable" is 0, then scaled with the TRAINING mean/sd.
    first_leg = transformers["first_leg"]
    scaler = first_leg.named_steps["standardscaler"]
    not_second_leg = pd.DataFrame({c: pd.array([pd.NA], dtype="Int64") for c in FIRST_LEG_FEATURES})
    np.testing.assert_allclose(first_leg.transform(not_second_leg)[0], (0 - scaler.mean_) / scaler.scale_)
    train_filled = X_train[FIRST_LEG_FEATURES].astype(float).fillna(0)
    np.testing.assert_allclose(scaler.mean_, train_filled.mean().to_numpy())


def test_new_format_stages_get_a_known_one_hot_column(fitted_lr, train_and_validation_inputs):
    _, _, X_val = train_and_validation_inputs
    preprocess = fitted_lr.named_steps["preprocess"]
    out = pd.DataFrame(preprocess.transform(X_val), columns=preprocess.get_feature_names_out())
    stage_columns = [c for c in out.columns if c.startswith("stage__")]
    assert (out[stage_columns].sum(axis=1) == 1).all()  # never an all-zero (unknown) row


# --- logistic regression ---------------------------------------------------------------

def test_logistic_regression_gives_valid_probabilities_in_label_order(fitted_lr, train_and_validation_inputs):
    _, _, X_val = train_and_validation_inputs
    assert list(fitted_lr.classes_) == ["A", "D", "H"]
    proba = predict_proba_frame(fitted_lr, X_val)
    assert list(proba.columns) == RESULT_CLASSES
    assert ((proba > 0) & (proba < 1)).all().all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_unknown_stage_at_prediction_time_raises(fitted_lr, train_and_validation_inputs):
    _, _, X_val = train_and_validation_inputs
    fixture = X_val.iloc[[0]].assign(stage="super_league")
    with pytest.raises(ValueError, match="super_league"):
        fitted_lr.predict_proba(fixture)


# --- tree models (part B) ---------------------------------------------------------------

def test_tree_preprocessors_do_not_scale_and_only_random_forest_imputes(train_and_validation_inputs):
    X_train, y_train, X_val = train_and_validation_inputs
    rf_pre = build_random_forest().named_steps["preprocess"].fit(X_train)
    hgb_pre = build_hist_gradient_boosting().named_steps["preprocess"].fit(X_train)
    for pre in (rf_pre, hgb_pre):
        steps = [step for _, branch, _ in pre.transformers_ if hasattr(branch, "steps") for _, step in branch.steps]
        assert not any(isinstance(step, StandardScaler) for step in steps)
    names = list(hgb_pre.get_feature_names_out())
    hgb_out = pd.DataFrame(hgb_pre.transform(X_val), columns=names)
    rf_out = pd.DataFrame(rf_pre.transform(X_val), columns=names)
    first_leg = "first_leg__first_leg_goal_diff"
    assert hgb_out[first_leg].isna().sum() == X_val["first_leg_goal_diff"].isna().sum() > 0  # NaN kept
    assert not rf_out.isna().any().any()
    assert (rf_out.loc[X_val["first_leg_goal_diff"].isna().to_numpy(), first_leg] == 0).all()


def test_hist_gradient_boosting_does_not_use_random_early_stopping():
    assert build_hist_gradient_boosting().named_steps["model"].early_stopping is False


@pytest.mark.parametrize("builder", [build_random_forest, build_hist_gradient_boosting])
def test_tree_models_give_valid_probabilities(builder, train_and_validation_inputs):
    X_train, y_train, X_val = train_and_validation_inputs
    model = builder(n_estimators=20) if builder is build_random_forest else builder(max_iter=20)
    proba = predict_proba_frame(model.fit(X_train, y_train), X_val)
    assert list(proba.columns) == RESULT_CLASSES
    assert ((proba >= 0) & (proba <= 1)).all().all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_candidate_models_include_the_prior_baseline():
    assert set(build_candidate_models()) == {"prior", "logistic_regression", "random_forest", "hist_gradient_boosting"}


# --- part C: flags, team identity, recency weights, saving ------------------------------

def test_every_boolean_feature_is_a_flag_and_every_flag_is_boolean(train_and_validation_inputs):
    """General rule: 0/1 features are never scaled, so each must be listed in FLAG_FEATURES."""
    X_train, _, _ = train_and_validation_inputs
    booleans = [c for c in ALLOWED_FEATURES if X_train[c].dtype == bool]
    assert sorted(booleans) == sorted(FLAG_FEATURES)


def test_logistic_regression_passes_flags_through_unscaled(fitted_lr, train_and_validation_inputs):
    X_train, _, _ = train_and_validation_inputs
    preprocess = fitted_lr.named_steps["preprocess"]
    out = pd.DataFrame(preprocess.transform(X_train), columns=preprocess.get_feature_names_out(), index=X_train.index)
    for flag in FLAG_FEATURES:
        pd.testing.assert_series_equal(out[f"flags__{flag}"], X_train[flag].astype(float), check_names=False)
    # continuous features are still standardised with training statistics
    assert out["continuous__elo_diff"].mean() == pytest.approx(0, abs=1e-9)
    assert out["continuous__elo_diff"].std(ddof=0) == pytest.approx(1)


def test_identity_one_hot_gives_an_unknown_team_an_all_zero_row():
    X = pd.DataFrame({"home_team": ["Real Madrid", "Ajax"], "away_team": ["Ajax", "Real Madrid"]})
    pre = build_preprocessor(IDENTITY_FEATURES).fit(X)
    out = pre.transform(pd.DataFrame({"home_team": ["Newcomer FC"], "away_team": ["Ajax"]}))
    names = list(pre.get_feature_names_out())
    assert out[0, [i for i, n in enumerate(names) if n.startswith("identity__home_team")]].sum() == 0
    assert out[0, names.index("identity__away_team_Ajax")] == 1


def test_season_age_weights_halve_every_half_life():
    rows = pd.DataFrame({"season": ["2020-21", "2021-22", "2022-23", "2022-23"]})
    np.testing.assert_allclose(season_age_weights(rows, half_life=2), [0.5, 0.5 ** 0.5, 1.0, 1.0])


def test_saved_model_loads_with_metadata_and_same_probabilities(fitted_lr, train_and_validation_inputs, tmp_path):
    _, _, X_val = train_and_validation_inputs
    model_path, metadata_path = save_model(fitted_lr, {"training_data_cutoff": "2024-06-01"}, "unit", tmp_path)
    loaded, metadata = load_model(model_path)
    assert metadata == {"training_data_cutoff": "2024-06-01"}
    assert metadata_path.read_text(encoding="utf-8").count("2024-06-01") == 1
    pd.testing.assert_frame_equal(predict_proba_frame(loaded, X_val), predict_proba_frame(fitted_lr, X_val))
