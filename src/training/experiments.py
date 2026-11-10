"""Milestone 7, part C: small hyperparameter grids, experiments and model selection.

Phase: Milestone 7 - Model training.

Previous stage: train.py builds the pipelines; evaluate.py scores them walk-forward.
Next stage: the selected pipeline is saved to models/; Milestone 8 scores it once on the
sealed test season, and predict.py (Milestone 9) uses it.

Every configuration is scored with walk-forward validation (config.WALK_FORWARD_SEASONS,
808 held-out matches). The test season is removed from the match table before any
feature is built, so no step here can read it.

Stages (the selection rule was fixed before any result, docs/data_notes.md):
1. Model grid (model_grid): LR C, RF min_samples_leaf, HGB depth x learning rate x
   iterations, all with the default features. select_grid_winner applies the rule:
   exclude share_over_90 > 2%, lowest pooled log loss, tie-break against the best other
   family (LR < RF < HGB), sanity check on the 2024-25 fold.
2. Experiments on the winner, each against its current Setting (choose_variant):
   Elo K x newcomer start, feature groups, team identity, recency weighting.
3. Combination of adopted changes (if more than one), final fit on every
   train/validation season, save_model.

Run: python -m src.training.experiments
"""

import platform
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial

import numpy as np
import pandas as pd
import sklearn

from src.config import (
    PROCESSED_DATA_DIR, PROJECT_ROOT, RANDOM_SEED, TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS,
    WALK_FORWARD_SEASONS,
)
from src.data.clean import load_clean_matches
from src.features.build_features import (
    ALLOWED_FEATURES, ELO_FEATURES, ELO_HOME_ADVANTAGE, ELO_K, ELO_SEASON_REGRESSION, ELO_START,
    FORM_FEATURES, IDENTITY_FEATURES, build_feature_table, empirical_newcomer_start,
)
from src.training.evaluate import (
    paired_bootstrap, per_match_log_loss, summarise_walk_forward, walk_forward_evaluate,
)
from src.training.split import split_features_target, walk_forward_folds
from src.training.train import (
    STAGE_GROUPS, build_baselines, build_hist_gradient_boosting, build_logistic_regression,
    build_random_forest, save_model, season_age_weights,
)

MAX_SHARE_OVER_90 = 0.02  # selection rule 1: more than 2% of matches above 0.9 -> excluded
FAMILY_SIMPLICITY = {"lr": 0, "rf": 1, "hgb": 2}  # selection rule 3: LR < RF < HGB
HGB_MIN_SAMPLES_LEAF = 50  # fixed (default 20): leaves average at least 50 matches
EARLIER_COMPARISONS = 3  # Parts A-B: LR C=1.0 with scaled flags, RF default, HGB default
USABLE_SEASONS = [*TRAIN_SEASONS, *VALIDATION_SEASONS]  # never TEST_SEASONS
FOCUS_SEASON = VALIDATION_SEASONS[-1]  # 2024-25: the sanity-check fold


def model_grid() -> dict:
    """Name -> builder (a partial taking `features`) for every grid configuration (14)."""
    grid = {f"lr_C{C}": partial(build_logistic_regression, C=C) for C in (0.1, 0.3, 1.0)}
    grid |= {f"rf_leaf{leaf}": partial(build_random_forest, min_samples_leaf=leaf) for leaf in (5, 20, 50)}
    for depth in (2, 3):
        for rate in (0.03, 0.05):
            for iterations in (100, 300):
                grid[f"hgb_d{depth}_lr{rate}_it{iterations}"] = partial(
                    build_hist_gradient_boosting, max_depth=depth, learning_rate=rate,
                    max_iter=iterations, min_samples_leaf=HGB_MIN_SAMPLES_LEAF)
    return grid


def family(name: str) -> str:
    """Model family of a grid name: "lr", "rf" or "hgb"."""
    return name.split("_")[0]


@dataclass(frozen=True)
class Setting:
    """Everything that defines a configuration besides the model's hyperparameters."""

    elo_k: float = ELO_K
    newcomer: str = "1500"  # "1500" (ELO_START) or "empirical" (measured per fold)
    features: tuple[str, ...] = tuple(ALLOWED_FEATURES)
    identity: bool = False  # add one-hot home_team / away_team
    half_life: float | None = None  # recency weighting in seasons; None = equal weights


class FeatureTables:
    """Feature tables rebuilt from the clean matches (test season removed), cached by Elo
    parameters, plus the walk-forward folds of a Setting."""

    def __init__(self, matches: pd.DataFrame):
        self.matches = matches[~matches["season"].isin(TEST_SEASONS)]
        if self.matches["season"].isin(TEST_SEASONS).any():
            raise ValueError("Test season rows reached the experiment tables")
        self._tables: dict[tuple, pd.DataFrame] = {}
        self.newcomer_starts: dict[tuple[float, str], float] = {}  # (K, fold season) -> start

    def table(self, k: float = ELO_K, newcomer_start: float | None = None) -> pd.DataFrame:
        key = (k, newcomer_start)
        if key not in self._tables:
            self._tables[key] = build_feature_table(self.matches, {"k": k, "newcomer_start": newcomer_start})
        return self._tables[key]

    def newcomer_start(self, k: float, seasons: list[str]) -> float:
        """Empirical newcomer start measured on `seasons` only (a fold's training seasons)."""
        return empirical_newcomer_start(self.matches, seasons, k=k)

    def folds(self, setting: Setting) -> list:
        if setting.newcomer == "1500":
            return walk_forward_folds(self.table(setting.elo_k), WALK_FORWARD_SEASONS)
        folds = []
        for season in WALK_FORWARD_SEASONS:
            start = self.newcomer_start(setting.elo_k, [s for s in USABLE_SEASONS if s < season])
            self.newcomer_starts[(setting.elo_k, season)] = start
            folds += walk_forward_folds(self.table(setting.elo_k, start), [season])
        return folds


def model_features(setting: Setting) -> list[str]:
    return [*setting.features, *(IDENTITY_FEATURES if setting.identity else [])]


def run_setting(label: str, builder, setting: Setting, tables: FeatureTables):
    """Walk-forward scores and predictions of builder(features) under `setting`."""
    weights = None if setting.half_life is None else partial(season_age_weights, half_life=setting.half_life)
    return walk_forward_evaluate(
        {label: builder(features=model_features(setting))}, tables.folds(setting),
        extra_features=IDENTITY_FEATURES if setting.identity else [], sample_weight=weights)


def select_grid_winner(summary: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Selection rules 1-4 on the grid summary (summarise_walk_forward output)."""
    candidates = summary.drop(index="prior").sort_values("val_log_loss")
    excluded = candidates.index[candidates["val_share_over_90"] > MAX_SHARE_OVER_90].tolist()
    eligible = candidates.drop(index=excluded)
    best = eligible.index[0]
    others = [name for name in eligible.index if family(name) != family(best)]
    result = {"best": best, "excluded": excluded, "runner_up": None, "bootstrap": None, "winner": best}
    if others:
        runner_up = others[0]
        boot = paired_bootstrap(predictions, best, runner_up)
        tie = boot["low"] <= 0 <= boot["high"]
        simpler = FAMILY_SIMPLICITY[family(runner_up)] < FAMILY_SIMPLICITY[family(best)]
        result |= {"runner_up": runner_up, "bootstrap": boot, "winner": runner_up if tie and simpler else best}
    focus = f"log_loss_{FOCUS_SEASON}"
    result["passes_sanity_check"] = bool(summary.loc[result["winner"], focus] < summary.loc["prior", focus])
    return result


def choose_variant(current: str, variants: dict[str, bool], summary: pd.DataFrame,
                   predictions: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Selection rule 5. variants: label -> True if simpler than the current setting.

    A variant qualifies if it is not excluded (rule 1) and either (a) its interval
    against the current setting is entirely below 0, or (b) it is simpler and the
    interval includes 0. The qualifying variant with the lowest pooled log loss wins;
    otherwise the current setting stays.
    """
    rows = []
    for label, simpler in variants.items():
        boot = paired_bootstrap(predictions, label, current)
        excluded = summary.loc[label, "val_share_over_90"] > MAX_SHARE_OVER_90
        better = boot["high"] < 0
        not_worse_and_simpler = simpler and boot["low"] <= 0 <= boot["high"]
        rows.append({"variant": label, "val_log_loss": summary.loc[label, "val_log_loss"],
                     f"log_loss_{FOCUS_SEASON}": summary.loc[label, f"log_loss_{FOCUS_SEASON}"],
                     "diff": boot["mean_diff"], "low": boot["low"], "high": boot["high"], "simpler": simpler,
                     "excluded": excluded, "qualifies": not excluded and (better or not_worse_and_simpler)})
    table = pd.DataFrame(rows).set_index("variant")
    qualifying = table[table["qualifies"]]
    return (qualifying["val_log_loss"].idxmin() if len(qualifying) else current), table


def unknown_team_mask(folds: list) -> pd.Series:
    """True for held-out matches with a club that never appears in that fold's training rows."""
    masks = []
    for _, fit_rows, held_out in folds:
        seen = set(fit_rows["home_team"]) | set(fit_rows["away_team"])
        masks.append(~(held_out["home_team"].isin(seen) & held_out["away_team"].isin(seen)))
    return pd.concat(masks)


def git_commit() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def fit_final(builder, setting: Setting, tables: FeatureTables) -> tuple:
    """Fit on every train/validation season (2012-13 to 2024-25). Returns (model, rows, elo)."""
    newcomer_start = None
    if setting.newcomer == "empirical":
        newcomer_start = tables.newcomer_start(setting.elo_k, USABLE_SEASONS)
    table = tables.table(setting.elo_k, newcomer_start)
    rows = table[table["season"].isin(USABLE_SEASONS)].sort_values("date", kind="stable")
    features = model_features(setting)
    X, y = split_features_target(rows, features, allowed=features)
    fit_params = {} if setting.half_life is None else {
        "model__sample_weight": season_age_weights(rows, setting.half_life)}
    model = builder(features=features).fit(X, y, **fit_params)
    elo = {"k": setting.elo_k, "home_advantage": ELO_HOME_ADVANTAGE, "season_regression": ELO_SEASON_REGRESSION,
           "start": ELO_START, "newcomer_start": newcomer_start if newcomer_start is not None else ELO_START,
           "newcomer_rule": setting.newcomer}
    return model, rows, elo


def main() -> None:
    pd.set_option("display.width", 250)
    tables = FeatureTables(load_clean_matches(PROCESSED_DATA_DIR / "matches.csv"))
    base = Setting()
    comparisons: list[str] = []

    # --- Stage 1: model grid ---------------------------------------------------------
    grid = model_grid()
    folds = tables.folds(base)
    scores, predictions = walk_forward_evaluate(
        {"prior": build_baselines()["prior"], **{name: b() for name, b in grid.items()}}, folds)
    comparisons += list(grid)
    summary = summarise_walk_forward(scores, focus_season=FOCUS_SEASON)
    print(f"STAGE 1: model grid, {len(grid)} configurations, {len(predictions) // (len(grid) + 1)} held-out matches")
    print(summary.round(4).to_string())
    pick = select_grid_winner(summary, predictions)
    boot = pick["bootstrap"]
    print(f"\nExcluded (share_over_90 > {MAX_SHARE_OVER_90:.0%}): {pick['excluded'] or 'none'}")
    print(f"Best: {pick['best']}; runner-up (other family): {pick['runner_up']}; "
          f"best - runner-up {boot['mean_diff']:+.4f} [{boot['low']:+.4f}, {boot['high']:+.4f}]")
    print(f"Grid winner: {pick['winner']}; beats prior on {FOCUS_SEASON}: {pick['passes_sanity_check']}")
    if not pick["passes_sanity_check"]:
        print("Sanity check failed: stopping before the experiments (selection rule 4).")
        return
    winner = pick["winner"]
    builder = grid[winner]

    # --- Stage 2: experiments on the grid winner ---------------------------------------
    experiments = {
        "Elo K x newcomer start": {
            f"elo_k{k:g}_{newcomer}": (replace(base, elo_k=k, newcomer=newcomer), False)
            for k in (15.0, 20.0, 30.0) for newcomer in ("1500", "empirical")
            if (k, newcomer) != (ELO_K, "1500")},
        "Feature groups": {
            "features_form": (replace(base, features=tuple(FORM_FEATURES)), True),
            "features_elo": (replace(base, features=tuple(ELO_FEATURES)), True),
            "features_form_elo": (replace(base, features=tuple(FORM_FEATURES + ELO_FEATURES)), True),
            "features_no_closed_doors": (replace(base, features=tuple(
                c for c in ALLOWED_FEATURES if c != "behind_closed_doors")), True)},
        "Team identity": {"identity": (replace(base, identity=True), False)},
        "Recency weighting": {f"recency_hl{h}": (replace(base, half_life=float(h)), False) for h in (3, 6)},
    }
    all_scores, all_predictions = [scores], [predictions]
    adopted: dict[str, tuple[str, Setting]] = {}
    for title, variants in experiments.items():
        for label, (setting, _) in variants.items():
            s, p = run_setting(label, builder, setting, tables)
            all_scores.append(s)
            all_predictions.append(p)
            comparisons.append(label)
        scores_all, predictions_all = pd.concat(all_scores), pd.concat(all_predictions)
        summary_all = summarise_walk_forward(scores_all, focus_season=FOCUS_SEASON)
        choice, table = choose_variant(winner, {k: v[1] for k, v in variants.items()}, summary_all, predictions_all)
        print(f"\nSTAGE 2: {title} (current: {winner}, pooled log loss {summary_all.loc[winner, 'val_log_loss']:.4f}, "
              f"{FOCUS_SEASON} {summary_all.loc[winner, f'log_loss_{FOCUS_SEASON}']:.4f})")
        print(table.round(4).to_string())
        print(f"-> {'keep current setting' if choice == winner else 'adopt ' + choice}")
        if choice != winner:
            adopted[title] = (choice, variants[choice][0])

    print("\nEmpirical newcomer starts (measured on each fold's training seasons):")
    for (k, season), start in sorted(tables.newcomer_starts.items()):
        print(f"  K={k:g}, fold {season}: {start:.1f}")
    unknown = unknown_team_mask(folds)
    by_unknown = predictions_all[predictions_all["model"].isin([winner, "identity"])].copy()
    by_unknown["unknown_team"] = unknown.reindex(by_unknown.index).to_numpy()
    by_unknown["loss"] = per_match_log_loss(by_unknown["result"], by_unknown[["H", "D", "A"]])
    print(f"\nTeam identity: held-out matches with a club unseen in that fold's training: {int(unknown.sum())} of {len(unknown)}")
    print(by_unknown.groupby(["model", "unknown_team"])["loss"].agg(["mean", "size"]).round(4).to_string())

    # --- Stage 3: combination and final model ------------------------------------------
    final_label, final_setting = winner, base
    if len(adopted) == 1:
        final_label, final_setting = next(iter(adopted.values()))
    elif len(adopted) > 1:
        combined = base
        for _, setting in adopted.values():
            changes = {f: getattr(setting, f) for f in Setting.__dataclass_fields__ if getattr(setting, f) != getattr(base, f)}
            combined = replace(combined, **changes)
        s, p = run_setting("combined", builder, combined, tables)
        comparisons.append("combined")
        summary_all = summarise_walk_forward(pd.concat([*all_scores, s]), focus_season=FOCUS_SEASON)
        only_removals = set(combined.features) < set(base.features) and replace(combined, features=base.features) == base
        choice, table = choose_variant(winner, {"combined": only_removals}, summary_all, pd.concat([*all_predictions, p]))
        print("\nSTAGE 3: combination of adopted changes")
        print(table.round(4).to_string())
        if choice == "combined":
            final_label, final_setting = "combined", combined
        else:
            best_single = min(adopted.values(), key=lambda item: summary_all.loc[item[0], "val_log_loss"])
            final_label, final_setting = best_single
            print(f"Combination does not qualify; using the best single adopted change: {final_label}")

    final_summary = summarise_walk_forward(pd.concat(all_scores + ([s] if final_label == "combined" else [])),
                                           focus_season=FOCUS_SEASON)
    final_row = final_summary.loc[final_label]
    prior_row = final_summary.loc["prior"]
    all_preds = pd.concat(all_predictions + ([p] if final_label == "combined" else []))
    vs_prior = paired_bootstrap(all_preds, final_label, "prior")
    passes = bool(final_row[f"log_loss_{FOCUS_SEASON}"] < prior_row[f"log_loss_{FOCUS_SEASON}"])
    total = EARLIER_COMPARISONS + len(comparisons)
    print(f"\nFINAL: {winner} with setting {final_setting}")
    print(f"pooled log loss {final_row['val_log_loss']:.4f} (prior {prior_row['val_log_loss']:.4f}); "
          f"{FOCUS_SEASON} {final_row[f'log_loss_{FOCUS_SEASON}']:.4f} (prior {prior_row[f'log_loss_{FOCUS_SEASON}']:.4f}); "
          f"vs prior {vs_prior['mean_diff']:+.4f} [{vs_prior['low']:+.4f}, {vs_prior['high']:+.4f}]")
    print(f"Comparisons on validation data: {total} ({EARLIER_COMPARISONS} in parts A-B, "
          f"{len(grid)} grid, {len(comparisons) - len(grid)} experiments)")
    if not passes:
        print("Final configuration does not beat the prior on the sanity-check fold: not saved.")
        return

    model, rows, elo = fit_final(builder, final_setting, tables)
    final_scores = final_summary.loc[[final_label]]
    metadata = {
        "name": "model_m7",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": winner,
        "family": family(winner),
        "hyperparameters": dict(builder.keywords),
        "estimator": repr(model.named_steps["model"]),
        "features": model_features(final_setting),
        "flags_unscaled": True,
        "stage_groups": STAGE_GROUPS,
        "elo": elo,
        "recency_half_life": final_setting.half_life,
        "training_seasons": sorted(rows["season"].unique()),
        "training_matches": len(rows),
        "training_data_cutoff": f"{rows['date'].max():%Y-%m-%d}",
        "test_seasons_sealed": TEST_SEASONS,
        "classes": [str(c) for c in model.classes_],
        "walk_forward": {
            "seasons": WALK_FORWARD_SEASONS,
            "held_out_matches": int(len(all_preds) // all_preds["model"].nunique()),
            "scores": {k: float(v) for k, v in final_scores.iloc[0].items()},
            "prior": {k: float(v) for k, v in prior_row.items()},
            "vs_prior_bootstrap": vs_prior,
            "grid_runner_up": pick["runner_up"],
            "grid_winner_vs_runner_up_bootstrap": pick["bootstrap"],
        },
        "selection_rule": "docs/data_notes.md, 'Selection rule (fixed 2026-11-10, before any Part C result)'",
        "validation_comparisons": total,
        "random_seed": RANDOM_SEED,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "git": git_commit(),
    }
    model_path, metadata_path = save_model(model, metadata, "model_m7")
    print(f"\nSaved {model_path} and {metadata_path}: fitted on {len(rows)} matches "
          f"({rows['season'].min()} to {rows['season'].max()}), cutoff {metadata['training_data_cutoff']}")


if __name__ == "__main__":
    main()
