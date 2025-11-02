"""Utilities for expected goal modelling and probability conversion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def _make_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",
        max_depth=6,
        learning_rate=0.05,
        max_iter=600,
        min_samples_leaf=25,
        l2_regularization=0.1,
        random_state=42,
    )


def train_goal_expectations(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Return leakage-safe expected goals for train/test splits."""

    oof = np.zeros(len(X_train))
    coverage = np.zeros(len(X_train), dtype=bool)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, val_idx in tscv.split(X_train):
        model = _make_regressor()
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        oof[val_idx] = preds
        coverage[val_idx] = True

    # For early rows not covered by TSCV, fill with global mean to avoid leakage
    if not coverage.all():
        fallback = max(y_train.mean(), 1e-6)
        oof[~coverage] = fallback

    final_model = _make_regressor()
    final_model.fit(X_train, y_train)
    test_pred = final_model.predict(X_test)

    oof = np.clip(oof, 1e-6, None)
    test_pred = np.clip(test_pred, 1e-6, None)

    metrics = {
        "mae": mean_absolute_error(y_train, oof),
        "rmse": mean_squared_error(y_train, oof) ** 0.5,
    }

    return oof, test_pred, metrics


def _default_goal_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select leakage-safe feature names for the expected goal regressors.
    Only pre-match engineered columns are used.
    """
    allowed_prefixes = (
        "feat_",
        "team_xg_rolling_",
        "opponent_xg_rolling_",
        "xg_diff_rolling_",
        "team_points_",
        "opponent_points_",
        "team_strength_",
        "opponent_strength_",
        "strength_",
        "points_",
        "rest_",
        "market_",
        "matchup_",
        "attack_",
        "defence_",
        "team_expected_",
        "opponent_expected_",
        "team_xg_diff_",
        "team_elo_",
        "opponent_elo_",
        "elo_",
    )
    allowed_exact = {
        "xg_form_mismatch",
        "team_match_number",
        "rest_days",
        "opponent_rest_days",
        "rest_days_diff",
        "strength_index_diff",
        "points_avg_5_diff",
        "expected_points_edge",
        "market_confidence_gap",
        "matchup_xg_diff_avg_3",
        "team_elo_pre",
        "opponent_elo_pre",
        "elo_diff",
        "elo_expected_score",
        "elo_edge",
    }

    numeric_cols = df.select_dtypes(include=["number", "float", "int"]).columns
    forbidden = {
        "team_goals",
        "opponent_goals",
        "team_xg",
        "opponent_xg",
        "goal_difference",
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
        "corners_for",
        "corners_against",
        "fouls_for",
        "fouls_against",
        "yellow_cards",
        "red_cards",
    }

    candidates: list[str] = []
    for col in numeric_cols:
        if col in forbidden or col.startswith("eg_prob_") or col.startswith("eg_lambda_"):
            continue
        if col.startswith(allowed_prefixes) or col in allowed_exact:
            candidates.append(col)
    return candidates


def _season_expected_goals(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    minimum_train: int = 60,
) -> np.ndarray:
    """Generate out-of-fold predictions for a single season."""
    n_samples = len(X)
    if n_samples == 0:
        return np.zeros(0, dtype=float)

    splits = min(n_splits, max(1, n_samples // 120))
    if splits < 2:
        splits = 2 if n_samples >= minimum_train + 1 else 1

    predictions = np.full(n_samples, np.nan, dtype=float)
    if splits >= 2 and n_samples > minimum_train:
        tscv = TimeSeriesSplit(n_splits=splits)
        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) < minimum_train:
                continue
            model = _make_regressor()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions[val_idx] = model.predict(X.iloc[val_idx])

    running_mean = y.expanding().mean().shift(1).fillna(y.mean())
    fallback = np.clip(running_mean.to_numpy(), 1e-6, None)
    predictions = np.where(np.isnan(predictions), fallback, predictions)
    predictions = np.clip(predictions, 1e-6, None)
    return predictions


def attach_expected_goal_features(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    *,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Enrich the provided dataset with expected-goal lambdas and outcome probabilities.

    The function performs a leakage-safe time-series CV within each season to
    estimate λ_for and λ_against, then converts them into win/draw/loss
    probabilities for downstream models.
    """
    if feature_columns is None:
        feature_columns = _default_goal_feature_columns(df)

    if not feature_columns:
        raise ValueError("No feature columns available for expected goal modelling.")

    working = df.sort_values(["season", "match_date", "team"]).copy()
    working["eg_lambda_team"] = np.nan
    working["eg_lambda_opponent"] = np.nan

    for season, season_idx in working.groupby("season").groups.items():
        season_slice = working.loc[season_idx]
        X_season = season_slice.loc[:, feature_columns].fillna(0.0)
        y_team = season_slice["team_goals"].astype(float)
        y_opp = season_slice["opponent_goals"].astype(float)

        team_played_mask = y_team.notna()
        opp_played_mask = y_opp.notna()

        lambda_team_vals = np.full(len(season_slice), np.nan, dtype=float)
        lambda_opp_vals = np.full(len(season_slice), np.nan, dtype=float)

        if team_played_mask.any():
            lt_played = _season_expected_goals(
                X_season.loc[team_played_mask],
                y_team.loc[team_played_mask],
                n_splits=n_splits,
            )
            lambda_team_vals[team_played_mask.to_numpy()] = lt_played

            future_mask = ~team_played_mask
            if future_mask.any():
                model = _make_regressor()
                model.fit(X_season.loc[team_played_mask], y_team.loc[team_played_mask])
                preds_future = model.predict(X_season.loc[future_mask])
                lambda_team_vals[future_mask.to_numpy()] = preds_future

        if opp_played_mask.any():
            lo_played = _season_expected_goals(
                X_season.loc[opp_played_mask],
                y_opp.loc[opp_played_mask],
                n_splits=n_splits,
            )
            lambda_opp_vals[opp_played_mask.to_numpy()] = lo_played

            future_mask = ~opp_played_mask
            if future_mask.any():
                model = _make_regressor()
                model.fit(X_season.loc[opp_played_mask], y_opp.loc[opp_played_mask])
                preds_future = model.predict(X_season.loc[future_mask])
                lambda_opp_vals[future_mask.to_numpy()] = preds_future

        if np.isnan(lambda_team_vals).all():
            mean_val = y_team.dropna().mean()
            lambda_team_vals[:] = np.clip(mean_val if pd.notna(mean_val) else 1.0, 1e-6, None)
        if np.isnan(lambda_opp_vals).all():
            mean_val = y_opp.dropna().mean()
            lambda_opp_vals[:] = np.clip(mean_val if pd.notna(mean_val) else 1.0, 1e-6, None)

        lambda_team_vals = np.clip(lambda_team_vals, 1e-6, None)
        lambda_opp_vals = np.clip(lambda_opp_vals, 1e-6, None)

        working.loc[season_idx, "eg_lambda_team"] = lambda_team_vals
        working.loc[season_idx, "eg_lambda_opponent"] = lambda_opp_vals
    working["eg_lambda_diff"] = working["eg_lambda_team"] - working["eg_lambda_opponent"]

    goal_probs = poisson_outcome_probabilities(
        working["eg_lambda_team"].to_numpy(),
        working["eg_lambda_opponent"].to_numpy(),
    )
    working["eg_prob_win"] = goal_probs[:, 0]
    working["eg_prob_draw"] = goal_probs[:, 1]
    working["eg_prob_loss"] = goal_probs[:, 2]

    if {"book_prob_win", "book_prob_draw", "book_prob_loss"}.issubset(working.columns):
        working["eg_prob_win_edge"] = working["eg_prob_win"] - working["book_prob_win"]
        working["eg_prob_draw_edge"] = working["eg_prob_draw"] - working["book_prob_draw"]
        working["eg_prob_loss_edge"] = working["eg_prob_loss"] - working["book_prob_loss"]

    aligned = (
        working.sort_index()[  # restore original ordering
            [
                "eg_lambda_team",
                "eg_lambda_opponent",
                "eg_lambda_diff",
                "eg_prob_win",
                "eg_prob_draw",
                "eg_prob_loss",
            ]
        ]
        .copy()
    )

    if "eg_prob_win_edge" in working.columns:
        aligned["eg_prob_win_edge"] = working.sort_index()["eg_prob_win_edge"]
    if "eg_prob_draw_edge" in working.columns:
        aligned["eg_prob_draw_edge"] = working.sort_index()["eg_prob_draw_edge"]
    if "eg_prob_loss_edge" in working.columns:
        aligned["eg_prob_loss_edge"] = working.sort_index()["eg_prob_loss_edge"]

    enriched = df.copy()
    for column in aligned.columns:
        enriched[column] = aligned[column].to_numpy()
    return enriched


def poisson_outcome_probabilities(
    lambda_for: np.ndarray,
    lambda_against: np.ndarray,
    max_goals: int = 8,
) -> np.ndarray:
    """Compute win/draw/loss probabilities under independent Poisson scoring."""

    factorial = [1]
    for i in range(1, max_goals + 1):
        factorial.append(factorial[-1] * i)

    n = len(lambda_for)
    probs = np.zeros((n, 3))

    for i in range(n):
        lam_f = float(lambda_for[i])
        lam_a = float(lambda_against[i])

        win = draw = loss = 0.0
        total = 0.0

        for h in range(max_goals + 1):
            p_h = np.exp(-lam_f) * lam_f ** h / factorial[h]
            for a in range(max_goals + 1):
                p_a = np.exp(-lam_a) * lam_a ** a / factorial[a]
                prob = p_h * p_a
                total += prob
                if h > a:
                    win += prob
                elif h == a:
                    draw += prob
                else:
                    loss += prob

        remainder = max(0.0, 1.0 - total)
        draw += remainder
        probs[i] = (win, draw, loss)

    return probs
