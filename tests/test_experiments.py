"""Tests for the Milestone 7 selection rule in src/training/experiments.py.

Small hand-made summaries and predictions: identical predictions give a bootstrap
interval of exactly [0, 0] (a tie); a model that is better on every match gives an
interval entirely below 0.
"""

import pandas as pd
import pytest

from src.training.experiments import FOCUS_SEASON, choose_variant, model_grid, select_grid_winner

RESULTS = ["H", "D", "A", "H", "A", "H"]


def predictions(model: str, p_actual: float) -> pd.DataFrame:
    """Every match gets probability p_actual on its actual result, the rest split evenly."""
    rows = []
    for i, result in enumerate(RESULTS):
        probs = {c: (1 - p_actual) / 2 for c in "HDA"}
        probs[result] = p_actual
        rows.append({"model": model, "season": FOCUS_SEASON, "result": result, **probs})
    return pd.DataFrame(rows, index=range(len(RESULTS)))


def summary(rows: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    """name -> (pooled log loss, share_over_90, focus-season log loss)."""
    return pd.DataFrame(
        {name: {"val_log_loss": ll, "val_share_over_90": share, f"log_loss_{FOCUS_SEASON}": focus}
         for name, (ll, share, focus) in rows.items()}).T.sort_values("val_log_loss")


def test_grid_has_the_agreed_fourteen_configurations():
    names = list(model_grid())
    assert len(names) == 14
    assert {n.split("_")[0] for n in names} == {"lr", "rf", "hgb"}


def test_tie_with_a_simpler_family_picks_the_simpler_model():
    s = summary({"prior": (1.04, 0, 1.0), "rf_leaf20": (0.95, 0, 0.95), "lr_C0.3": (0.96, 0, 0.93)})
    p = pd.concat([predictions("rf_leaf20", 0.5), predictions("lr_C0.3", 0.5), predictions("prior", 0.4)])
    pick = select_grid_winner(s, p)
    assert pick["best"] == "rf_leaf20" and pick["runner_up"] == "lr_C0.3"
    assert pick["bootstrap"]["low"] == pick["bootstrap"]["high"] == 0
    assert pick["winner"] == "lr_C0.3" and pick["passes_sanity_check"]


def test_clear_winner_is_kept_even_if_more_complex():
    s = summary({"prior": (1.04, 0, 1.0), "rf_leaf20": (0.80, 0, 0.8), "lr_C0.3": (0.96, 0, 0.93)})
    p = pd.concat([predictions("rf_leaf20", 0.6), predictions("lr_C0.3", 0.4), predictions("prior", 0.4)])
    pick = select_grid_winner(s, p)
    assert pick["bootstrap"]["high"] < 0
    assert pick["winner"] == "rf_leaf20"


def test_overconfident_model_is_excluded_and_sanity_check_can_fail():
    s = summary({"prior": (1.04, 0, 0.9), "hgb_d3": (0.70, 0.05, 0.7), "lr_C0.3": (0.96, 0, 0.95)})
    p = pd.concat([predictions("hgb_d3", 0.7), predictions("lr_C0.3", 0.4), predictions("prior", 0.4)])
    pick = select_grid_winner(s, p)
    assert pick["excluded"] == ["hgb_d3"]
    assert pick["winner"] == "lr_C0.3" and pick["runner_up"] is None
    assert not pick["passes_sanity_check"]  # 0.95 is not below the prior's 0.9


def test_choose_variant_rules_a_and_b():
    s = summary({"current": (0.95, 0, 0.9), "better": (0.90, 0, 0.9), "same_simpler": (0.95, 0, 0.9),
                 "same_complex": (0.95, 0, 0.9), "worse_simpler": (1.00, 0, 1.0)})
    p = pd.concat([predictions("current", 0.5), predictions("better", 0.6), predictions("same_simpler", 0.5),
                   predictions("same_complex", 0.5), predictions("worse_simpler", 0.4)])
    choice, table = choose_variant(
        "current", {"better": False, "same_simpler": True, "same_complex": False, "worse_simpler": True}, s, p)
    assert table["qualifies"].to_dict() == {
        "better": True, "same_simpler": True, "same_complex": False, "worse_simpler": False}
    assert choice == "better"  # lowest pooled log loss among qualifying variants


def test_choose_variant_keeps_current_when_nothing_qualifies():
    s = summary({"current": (0.95, 0, 0.9), "same_complex": (0.95, 0, 0.9)})
    p = pd.concat([predictions("current", 0.5), predictions("same_complex", 0.5)])
    choice, _ = choose_variant("current", {"same_complex": False}, s, p)
    assert choice == "current"
