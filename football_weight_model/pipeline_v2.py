"""
Fixture-level modelling pipeline (Football Weight Model v3)
Implements Poisson goal models, gradient boosting classifier,
per-class isotonic calibration, and diagnostics-only bookmaker comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError, lstsq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from .calibration import apply_isotonic, fit_isotonic_per_class
from .fixture_features import build_fixture_dataset
from .market_compare import compare_model_vs_market
from .validation import evaluate_metrics, walk_forward_splits

RESULT_TO_INT = {"win": 0, "draw": 1, "loss": 2}
INT_TO_RESULT = {v: k for k, v in RESULT_TO_INT.items()}

GOAL_REGRESSOR_PARAMS = dict(
    loss="poisson",
    max_depth=6,
    learning_rate=0.05,
    max_iter=800,
    l2_regularization=0.1,
    min_samples_leaf=20,
    random_state=42,
)

CLASSIFIER_PARAMS = dict(
    max_depth=6,
    learning_rate=0.05,
    max_iter=600,
    min_samples_leaf=40,
    l2_regularization=0.1,
    class_weight={0: 1.0, 1: 0.7, 2: 1.0},
    random_state=21,
)

BOOKMAKER_COLUMN_PREFIXES = ("book_", "home_odds", "away_odds", "market_")
FEATURE_MIN_CORR = 0.02
VIF_THRESHOLD = 10.0
BLEND_WEIGHT_CLASSIFIER = 0.60
BLEND_WEIGHT_POISSON = 0.40
FORECAST_OUTPUT_COLUMNS = [
    "season",
    "match_date",
    "home_team",
    "away_team",
    "match_id",
    "home_odds_win",
    "home_odds_draw",
    "home_odds_loss",
    "final_p_home",
    "final_p_draw",
    "final_p_away",
    "confidence",
    "book_prob_home",
    "book_prob_draw",
    "book_prob_away",
]


def _create_fixture_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["season"].astype(str)
        + "|"
        + df["match_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + df["home_team"]
        + "|"
        + df["away_team"]
    )


def _implied_probs(odds: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / odds.replace(0.0, np.nan)
    overround = inv.sum(axis=1)
    implied = inv.div(overround, axis=0)
    implied.columns = ["book_prob_home", "book_prob_draw", "book_prob_away"]
    return implied.fillna(np.nan)


def _filter_feature_columns(feature_columns: List[str]) -> List[str]:
    filtered = []
    for col in feature_columns:
        if any(col.startswith(prefix) for prefix in BOOKMAKER_COLUMN_PREFIXES):
            continue
        if col in {"home_goals", "away_goals", "label", "fixture_id"}:
            continue
        filtered.append(col)
    return filtered


def _correlation_filter(X: pd.DataFrame, y_numeric: np.ndarray) -> List[str]:
    kept: List[str] = []
    for col in X.columns:
        series = X[col].to_numpy(dtype=float)
        if np.allclose(series.std(), 0.0):
            continue
        corr = np.corrcoef(series, y_numeric)[0, 1]
        if np.isnan(corr):
            continue
        if abs(corr) >= FEATURE_MIN_CORR:
            kept.append(col)
    return kept


def _compute_vif_matrix(values: np.ndarray) -> np.ndarray:
    n_features = values.shape[1]
    vifs = np.zeros(n_features, dtype=float)
    for i in range(n_features):
        y = values[:, i]
        X = np.delete(values, i, axis=1)
        if X.size == 0:
            vifs[i] = 1.0
            continue
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
        except LinAlgError:
            vifs[i] = np.inf
            continue
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot <= 1e-9:
            vifs[i] = np.inf
            continue
        r_squared = 1 - ss_res / ss_tot
        if r_squared >= 1.0:
            vifs[i] = np.inf
        else:
            vifs[i] = 1.0 / (1.0 - r_squared)
    return vifs


def _vif_pruning(X: pd.DataFrame) -> List[str]:
    cols = list(X.columns)
    values = X.to_numpy(dtype=float)
    while len(cols) > 1:
        vifs = _compute_vif_matrix(values)
        max_idx = int(np.nanargmax(vifs))
        if vifs[max_idx] <= VIF_THRESHOLD or np.isinf(vifs[max_idx]):
            break
        cols.pop(max_idx)
        values = np.delete(values, max_idx, axis=1)
    return cols


def _prepare_training_frames(fixtures: pd.DataFrame, feature_columns: List[str]):
    played = fixtures[fixtures["label"].notna()].copy()
    played = played.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)
    X = played[feature_columns].fillna(0.0)
    y_home = played["home_goals"].astype(float)
    y_away = played["away_goals"].astype(float)
    y_outcome = played["label"].map(RESULT_TO_INT).astype(int)
    return played, X, y_home, y_away, y_outcome


def _fit_goal_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**GOAL_REGRESSOR_PARAMS)


def _fit_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**CLASSIFIER_PARAMS)


def _prepare_dataset(team_dataset_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    team_df = pd.read_csv(team_dataset_path, parse_dates=["match_date"])
    metadata_path = team_dataset_path.with_name("epl_unified_feature_columns.json")
    if metadata_path.exists() and {"home_team", "away_team"}.issubset(team_df.columns):
        feature_columns = json.loads(metadata_path.read_text())
        fixtures = team_df.copy()
    else:
        fixtures, feature_columns = build_fixture_dataset(team_df)
    fixtures["fixture_id"] = _create_fixture_id(fixtures)
    if fixtures["fixture_id"].duplicated().any():
        dupes = fixtures[fixtures["fixture_id"].duplicated()]["fixture_id"].unique()
        raise ValueError(f"Duplicate fixture identifiers detected: {dupes[:5]}")
    return fixtures, feature_columns


def _select_features(fixtures: pd.DataFrame, feature_columns: List[str]) -> List[str]:
    filtered = _filter_feature_columns(feature_columns)
    played = fixtures[fixtures["label"].notna()].dropna(subset=["home_goals", "away_goals"])
    if played.empty:
        return filtered
    X = played[filtered].fillna(0.0)
    y_numeric = played["label"].map(RESULT_TO_INT).to_numpy(dtype=float)
    corr_kept = _correlation_filter(X, y_numeric)
    if not corr_kept:
        corr_kept = filtered
    X_corr = X[corr_kept]
    vif_kept = _vif_pruning(X_corr)
    return vif_kept


def _scale_features(X_train: pd.DataFrame, X_valid: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler, scaler.transform(X_train), scaler.transform(X_valid)


def _blend_probabilities(poisson_probs: np.ndarray, clf_probs: np.ndarray) -> np.ndarray:
    blended = (
        BLEND_WEIGHT_CLASSIFIER * clf_probs
        + BLEND_WEIGHT_POISSON * poisson_probs
    )
    blended = blended / blended.sum(axis=1, keepdims=True)
    return blended


def run_pipeline(
    team_dataset_path: Path,
    outputs_dir: Path,
    forecast_horizon_days: int = 7,
) -> Dict[str, pd.DataFrame]:
    fixtures, feature_columns_raw = _prepare_dataset(team_dataset_path)
    fixtures.sort_values(["season", "match_date", "home_team"], inplace=True)
    feature_columns = _select_features(fixtures, feature_columns_raw)

    played, X_full, y_home, y_away, y_outcome = _prepare_training_frames(fixtures, feature_columns)
    if played.empty:
        raise ValueError("No completed matches available to train the model.")

    n_samples = len(played)
    oof_poisson = np.zeros((n_samples, 3))
    oof_classifier_raw = np.zeros((n_samples, 3))
    fold_metrics: List[Dict[str, float]] = []

    splits = walk_forward_splits(played)
    for split in splits:
        train_idx = split.train_index
        valid_idx = split.valid_index

        X_train_raw = X_full.iloc[train_idx].fillna(0.0)
        X_valid_raw = X_full.iloc[valid_idx].fillna(0.0)

        scaler, X_train, X_valid = _scale_features(X_train_raw, X_valid_raw)

        model_home = _fit_goal_regressor()
        model_away = _fit_goal_regressor()
        model_home.fit(X_train, y_home.iloc[train_idx])
        model_away.fit(X_train, y_away.iloc[train_idx])

        lam_train_home = np.clip(model_home.predict(X_train), 0.05, 4.5)
        lam_train_away = np.clip(model_away.predict(X_train), 0.05, 4.5)
        lam_valid_home = np.clip(model_home.predict(X_valid), 0.05, 4.5)
        lam_valid_away = np.clip(model_away.predict(X_valid), 0.05, 4.5)

        goal_probs_valid = lambdas_to_probs(lam_valid_home, lam_valid_away)

        clf_train_features = np.hstack([
            X_train,
            lam_train_home.reshape(-1, 1),
            lam_train_away.reshape(-1, 1),
            (lam_train_home - lam_train_away).reshape(-1, 1),
        ])
        clf_valid_features = np.hstack([
            X_valid,
            lam_valid_home.reshape(-1, 1),
            lam_valid_away.reshape(-1, 1),
            (lam_valid_home - lam_valid_away).reshape(-1, 1),
        ])

        classifier = _fit_classifier()
        classifier.fit(clf_train_features, y_outcome.iloc[train_idx])
        clf_valid_raw = classifier.predict_proba(clf_valid_features)

        calibrator = fit_isotonic_per_class(clf_valid_raw, y_outcome.iloc[valid_idx].to_numpy())
        clf_valid_cal = apply_isotonic(clf_valid_raw, calibrator)

        blended_valid = _blend_probabilities(goal_probs_valid, clf_valid_cal)

        oof_poisson[valid_idx] = goal_probs_valid
        oof_classifier_raw[valid_idx] = clf_valid_raw

        metrics = evaluate_metrics(y_outcome.iloc[valid_idx].to_numpy(), blended_valid)
        metrics.update(
            {
                "fold": f"{'+'.join(split.train_seasons)}→{split.valid_season}",
                "train_matches": len(train_idx),
                "valid_matches": len(valid_idx),
            }
        )
        fold_metrics.append(metrics)

    final_calibrator = fit_isotonic_per_class(oof_classifier_raw, y_outcome.to_numpy())

    scaler_all = StandardScaler()
    X_full_scaled = scaler_all.fit_transform(X_full.fillna(0.0))

    final_home_model = _fit_goal_regressor()
    final_away_model = _fit_goal_regressor()
    final_home_model.fit(X_full_scaled, y_home)
    final_away_model.fit(X_full_scaled, y_away)

    lam_full_home = np.clip(final_home_model.predict(X_full_scaled), 0.05, 4.5)
    lam_full_away = np.clip(final_away_model.predict(X_full_scaled), 0.05, 4.5)

    clf_full_features = np.hstack([
        X_full_scaled,
        lam_full_home.reshape(-1, 1),
        lam_full_away.reshape(-1, 1),
        (lam_full_home - lam_full_away).reshape(-1, 1),
    ])
    final_classifier = _fit_classifier()
    final_classifier.fit(clf_full_features, y_outcome)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = outputs_dir / "model_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_home_model, artifacts_dir / "goals_poisson_home.pkl")
    joblib.dump(final_away_model, artifacts_dir / "goals_poisson_away.pkl")
    joblib.dump(final_classifier, artifacts_dir / "outcome_classifier.pkl")
    joblib.dump(final_calibrator, artifacts_dir / "calibrators.pkl")
    joblib.dump(scaler_all, artifacts_dir / "feature_scaler.pkl")
    (artifacts_dir / "selected_features.json").write_text(json.dumps(feature_columns))

    full_feature_matrix = fixtures[feature_columns].fillna(0.0)
    scaled_all = scaler_all.transform(full_feature_matrix)
    lambda_home_all = np.clip(final_home_model.predict(scaled_all), 0.05, 4.5)
    lambda_away_all = np.clip(final_away_model.predict(scaled_all), 0.05, 4.5)
    poisson_probs_all = lambdas_to_probs(lambda_home_all, lambda_away_all)

    clf_features_all = np.hstack([
        scaled_all,
        lambda_home_all.reshape(-1, 1),
        lambda_away_all.reshape(-1, 1),
        (lambda_home_all - lambda_away_all).reshape(-1, 1),
    ])
    clf_raw_all = final_classifier.predict_proba(clf_features_all)
    clf_cal_all = apply_isotonic(clf_raw_all, final_calibrator)

    blended_all = _blend_probabilities(poisson_probs_all, clf_cal_all)
    confidence = blended_all.max(axis=1) - blended_all.min(axis=1)

    odds_cols = ["home_odds_win", "home_odds_draw", "home_odds_loss"]
    implied = _implied_probs(fixtures[odds_cols])
    fixtures = pd.concat([fixtures.reset_index(drop=True), implied.reset_index(drop=True)], axis=1)

    final_predictions = fixtures.copy()
    final_predictions["p_poisson_home"] = poisson_probs_all[:, 0]
    final_predictions["p_poisson_draw"] = poisson_probs_all[:, 1]
    final_predictions["p_poisson_away"] = poisson_probs_all[:, 2]
    final_predictions["p_classifier_home"] = clf_cal_all[:, 0]
    final_predictions["p_classifier_draw"] = clf_cal_all[:, 1]
    final_predictions["p_classifier_away"] = clf_cal_all[:, 2]
    final_predictions["final_p_home"] = blended_all[:, 0]
    final_predictions["final_p_draw"] = blended_all[:, 1]
    final_predictions["final_p_away"] = blended_all[:, 2]
    final_predictions["confidence"] = confidence
    final_predictions["prediction"] = blended_all.argmax(axis=1)
    final_predictions["prediction"] = final_predictions["prediction"].map(INT_TO_RESULT)
    final_predictions["correct"] = np.where(
        final_predictions["label"].notna(),
        final_predictions["prediction"] == final_predictions["label"],
        np.nan,
    )

    final_predictions.to_csv(outputs_dir / "final_predictions.csv", index=False)

    played_final = final_predictions[final_predictions["label"].notna()].copy()
    season_rows = []
    for season, group in played_final.groupby("season"):
        y_true = group["label"].map(RESULT_TO_INT).to_numpy()
        probs = group[["final_p_home", "final_p_draw", "final_p_away"]].to_numpy()
        metrics = evaluate_metrics(y_true, probs)
        metrics["season"] = season
        season_rows.append(metrics)
    season_metrics = pd.DataFrame(season_rows).sort_values("season")
    season_metrics["accuracy_change_from_previous"] = season_metrics["accuracy"].diff()
    season_metrics.rename(
        columns={
            "accuracy": "model_accuracy",
            "brier": "model_brier_score",
            "logloss": "model_logloss",
            "ece": "model_ece",
        },
        inplace=True,
    )
    season_metrics.to_csv(outputs_dir / "season_metrics.csv", index=False)

    validation_metrics = pd.DataFrame(fold_metrics)
    validation_metrics.to_csv(outputs_dir / "validation_metrics.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "validation_accuracy", "value": validation_metrics["accuracy"].mean()},
            {"metric": "validation_brier", "value": validation_metrics["brier"].mean()},
            {"metric": "validation_logloss", "value": validation_metrics["logloss"].mean()},
            {"metric": "validation_ece", "value": validation_metrics["ece"].mean()},
        ]
    )
    summary.to_csv(outputs_dir / "summary.csv", index=False)

    future_mask = final_predictions["home_goals"].isna()
    future = final_predictions[future_mask].copy()
    forecast = future[FORECAST_OUTPUT_COLUMNS].sort_values("match_date")
    forecast.to_csv(outputs_dir / "forecast_clean.csv", index=False)
    final_predictions.to_csv(outputs_dir / "forecast_raw.csv", index=False)

    comparison_input = forecast.rename(
        columns={
            "final_p_home": "p_home_win",
            "final_p_draw": "p_draw",
            "final_p_away": "p_away_win",
        }
    )
    comparison = compare_model_vs_market(comparison_input)
    comparison.to_csv(outputs_dir / "diagnostics_market_gap.csv", index=False)

    readme = outputs_dir / "README_forecast.md"
    readme.write_text(
        "# Fixture Forecast Outputs

"
        "- `forecast_raw.csv`: model probabilities for all fixtures (home perspective).
"
        "- `forecast_clean.csv`: next-window forecasts with blended probabilities.
"
        "- `diagnostics_market_gap.csv`: comparison between model and bookmaker probabilities (bookmaker data used for diagnostics only).
"
        "- `summary.csv`, `validation_metrics.csv`, `season_metrics.csv`: walk-forward performance reports.
"
    )

    return {
        "summary": summary,
        "validation_metrics": validation_metrics,
        "season_metrics": season_metrics,
        "predictions": final_predictions,
        "forecast": forecast,
    }
