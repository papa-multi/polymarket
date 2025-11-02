"""
Football Weight Model v3 (xG-centric redesign).

This pipeline rebuilds the modelling workflow around expected goals,
Poisson/Skellam match probabilities, rolling-window validation, and
stacked ensembles with segmented calibration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from .calibration import IsotonicCalibrators, apply_isotonic, fit_isotonic_per_class
from .market_compare import compare_model_vs_market
from .validation import evaluate_metrics, expected_calibration_error

RESULT_TO_INT = {"win": 0, "draw": 1, "loss": 2}
INT_TO_RESULT = {v: k for k, v in RESULT_TO_INT.items()}

TRAIN_WINDOW = 200
VALID_WINDOW = 50
MIN_VALID_MATCHES = 30
MIN_CALIBRATION_SAMPLES = 20
FAVORITE_THRESHOLD = 0.6
UNDERDOG_THRESHOLD = 0.3
POISSON_MAX_GOALS = 8

REGRESSOR_PARAMS = dict(
    learning_rate=0.05,
    max_iter=700,
    max_depth=6,
    min_samples_leaf=25,
    l2_regularization=0.05,
    random_state=42,
)

GRADIENT_CLASSIFIER_PARAMS = dict(
    learning_rate=0.06,
    max_iter=550,
    max_depth=5,
    min_samples_leaf=25,
    l2_regularization=0.08,
    class_weight={0: 1.0, 1: 1.2, 2: 1.0},
    random_state=21,
)

FORM_LOGISTIC_PARAMS = dict(
    solver="lbfgs",
    max_iter=400,
    C=0.5,
)

META_STACK_PARAMS = dict(
    solver="lbfgs",
    max_iter=600,
    C=0.3,
)

FORM_PATTERNS = (
    "form_points",
    "win_streak",
    "unbeaten",
    "trend",
    "rest_days",
    "dynamic_strength",
    "strength_edge",
)

SEGMENT_NAMES = ("favorite", "balanced", "underdog")
MIN_SHRINK = {"favorite": 0.06, "balanced": 0.06, "underdog": 0.06, "global": 0.06}
EXCLUDED_FEATURE_PATTERNS = ("corners", "fouls", "discipline", "cards")
DRAW_DAMPING = 1.0
FINAL_SHRINK_DEFAULT = 0.2

VARIANCE_PARAMS = dict(
    learning_rate=0.05,
    max_iter=400,
    max_depth=4,
    min_samples_leaf=20,
    l2_regularization=0.05,
    random_state=42,
)

LAMBDA_KEYWORDS = (
    "attack_rating",
    "defence_rating",
    "strength_index",
    "form_factor",
    "xg_for_ewm",
    "xg_against_ewm",
    "xg_trend",
    "form_points",
    "rest_days",
    "match_style_index",
    "points_trend",
    "dynamic_strength",
)

VARIANCE_KEYWORDS = (
    "form_factor",
    "match_style_index",
    "xg_overperformance",
    "xg_conversion",
    "rest_days",
    "strength_index",
    "pred_xg_total",
    "pred_xg_diff",
    "attack_rating",
    "defence_rating",
)


def _dc_adjustment(lam_h: float, lam_a: float, rho: float, gh: int, ga: int) -> float:
    if gh == 0 and ga == 0:
        return max(1.0 - lam_h * lam_a * rho, 1e-6)
    if gh == 0 and ga == 1:
        return max(1.0 + lam_a * rho, 1e-6)
    if gh == 1 and ga == 0:
        return max(1.0 + lam_h * rho, 1e-6)
    if gh == 1 and ga == 1:
        return max(1.0 - rho, 1e-6)
    return 1.0


def _apply_dixon_coles_matrix(lam_h: float, lam_a: float, rho: float, max_goals: int) -> np.ndarray:
    lam_h = max(lam_h, 1e-4)
    lam_a = max(lam_a, 1e-4)
    home_pmf = _poisson_pmf_vector(lam_h, max_goals)
    away_pmf = _poisson_pmf_vector(lam_a, max_goals)
    matrix = np.outer(home_pmf, away_pmf)
    for gh in range(min(2, max_goals + 1)):
        for ga in range(min(2, max_goals + 1)):
            matrix[gh, ga] *= _dc_adjustment(lam_h, lam_a, rho, gh, ga)
    return matrix


def _fit_dc_rho(lambda_home: np.ndarray, lambda_away: np.ndarray, goals_home: np.ndarray, goals_away: np.ndarray) -> float:
    grid = np.linspace(-0.2, 0.2, 41)
    best_rho = 0.0
    best_ll = -np.inf
    for rho in grid:
        ll = 0.0
        for lam_h, lam_a, gh, ga in zip(lambda_home, lambda_away, goals_home, goals_away):
            gh_i = int(min(gh, POISSON_MAX_GOALS))
            ga_i = int(min(ga, POISSON_MAX_GOALS))
            matrix = _apply_dixon_coles_matrix(lam_h, lam_a, rho, POISSON_MAX_GOALS)
            prob = matrix[gh_i, ga_i]
            if prob <= 0:
                prob = 1e-12
            ll += np.log(prob)
        if ll > best_ll:
            best_ll = ll
            best_rho = float(rho)
    return best_rho


@dataclass
class RollingSplit:
    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray


@dataclass
class RollingSplitInfo:
    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    train_matches: int
    valid_matches: int
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    @property
    def label(self) -> str:
        start = self.valid_start.strftime("%Y-%m-%d")
        end = self.valid_end.strftime("%Y-%m-%d")
        return f"{start}→{end}"


def _implied_probs(odds: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / odds.replace(0.0, np.nan)
    overround = inv.sum(axis=1)
    implied = inv.div(overround, axis=0)
    implied.columns = ["book_prob_home", "book_prob_draw", "book_prob_away"]
    return implied.fillna(np.nan)


def _poisson_pmf_vector(lam: float, max_goals: int) -> np.ndarray:
    lam = max(lam, 1e-4)
    pmf = np.zeros(max_goals + 1, dtype=float)
    pmf[0] = np.exp(-lam)
    for k in range(1, max_goals + 1):
        pmf[k] = pmf[k - 1] * lam / k
    return pmf


def _poisson_match_probs(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float = 0.0,
    max_goals: int = POISSON_MAX_GOALS,
) -> np.ndarray:
    probs = np.zeros((len(lambda_home), 3), dtype=float)
    for idx, (lam_h, lam_a) in enumerate(zip(lambda_home, lambda_away)):
        matrix = _apply_dixon_coles_matrix(float(lam_h), float(lam_a), rho, max_goals)
        draw = np.trace(matrix)
        home_win = np.triu(matrix, k=1).sum()
        away_win = np.tril(matrix, k=-1).sum()
        covered = home_win + draw + away_win
        if covered < 0.999:
            draw += 1.0 - covered

        total = home_win + draw + away_win
        if total <= 0.0:
            probs[idx] = [1.0, 0.0, 0.0]
            continue
        probs[idx] = [home_win / total, draw / total, away_win / total]
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _augment_with_xg(features: pd.DataFrame, home_xg: np.ndarray, away_xg: np.ndarray) -> pd.DataFrame:
    augmented = features.copy()
    augmented["pred_home_xg"] = home_xg
    augmented["pred_away_xg"] = away_xg
    augmented["pred_xg_diff"] = home_xg - away_xg
    return augmented


def _select_form_features(columns: Iterable[str]) -> List[str]:
    selected = [col for col in columns if any(pattern in col for pattern in FORM_PATTERNS)]
    if not selected:
        return list(columns)
    return selected


def _prune_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    if df.empty:
        return list(df.columns)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = set()
    for column in upper.columns:
        if any(upper[column] > threshold):
            drop_cols.add(column)
    return [col for col in df.columns if col not in drop_cols]


def _assign_segments(probas: np.ndarray) -> np.ndarray:
    segments = np.full(probas.shape[0], "balanced", dtype=object)
    max_prob = probas.max(axis=1)
    segments[max_prob >= FAVORITE_THRESHOLD] = "favorite"
    segments[max_prob <= UNDERDOG_THRESHOLD] = "underdog"
    return segments


def _select_columns_by_keywords(columns: Iterable[str], keywords: Tuple[str, ...]) -> List[str]:
    selected = [col for col in columns if any(key in col for key in keywords)]
    return selected if selected else list(columns)


def _fit_segmented_calibrators(probas: np.ndarray, y: np.ndarray, segments: np.ndarray) -> Dict[str, IsotonicCalibrators | None]:
    calibrators: Dict[str, IsotonicCalibrators | None] = {}
    for seg in SEGMENT_NAMES:
        mask = segments == seg
        if mask.sum() >= MIN_CALIBRATION_SAMPLES:
            calibrators[seg] = fit_isotonic_per_class(probas[mask], y[mask])
        else:
            calibrators[seg] = None
    calibrators["global"] = fit_isotonic_per_class(probas, y)
    return calibrators


def _compute_shrink_factors(probas: np.ndarray, y: np.ndarray, segments: np.ndarray) -> Dict[str, float]:
    shrink: Dict[str, float] = {}
    grid = np.linspace(0.0, 0.8, 17)
    for seg in SEGMENT_NAMES:
        mask = segments == seg
        if not mask.any():
            shrink[seg] = 0.0
            continue
        best_alpha = 0.0
        best_score = np.inf
        for alpha in grid:
            adjusted = (1.0 - alpha) * probas[mask] + alpha / 3.0
            score = log_loss(y[mask], adjusted, labels=[0, 1, 2])
            score += expected_calibration_error(y[mask], adjusted) * 2.0
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
        shrink[seg] = min(best_alpha, 0.2)
    best_alpha = 0.0
    best_score = np.inf
    for alpha in grid:
        adjusted = (1.0 - alpha) * probas + alpha / 3.0
        score = log_loss(y, adjusted, labels=[0, 1, 2])
        score += expected_calibration_error(y, adjusted) * 2.0
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    shrink["global"] = min(best_alpha, 0.2)
    return shrink


def _recommendation_flags(
    probas: np.ndarray, min_confidence: float = 0.65, min_gap: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_prob = probas.max(axis=1)
    sorted_probs = np.sort(probas, axis=1)
    second_prob = sorted_probs[:, -2]
    flags = (max_prob >= min_confidence) & ((max_prob - second_prob) >= min_gap)
    predicted_class = probas.argmax(axis=1)
    return flags, max_prob, second_prob, predicted_class


def _apply_draw_guardrails(
    probas: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    variance: Optional[np.ndarray] = None,
) -> np.ndarray:
    adjusted = probas.copy()
    delta = np.abs(lambda_home - lambda_away)
    total = lambda_home + lambda_away
    for idx in range(len(adjusted)):
        if not np.isfinite(delta[idx]):
            continue
        if delta[idx] >= 0.6:
            max_draw = 0.36
        elif delta[idx] < 0.3:
            max_draw = 0.48
            if total[idx] < 2.4:
                max_draw = 0.55
        else:
            max_draw = 0.40

        if variance is not None:
            var_val = float(variance[idx])
            if var_val >= 3.0:
                max_draw *= 0.93
            elif var_val <= 1.0:
                max_draw = min(max_draw * 1.08, 0.58)
        current_draw = adjusted[idx, 1]
        if current_draw <= max_draw:
            continue
        reduction = current_draw - max_draw
        adjusted[idx, 1] = max_draw
        redistribution = adjusted[idx, [0, 2]].sum()
        if redistribution <= 1e-6:
            adjusted[idx, 0] += reduction * 0.5
            adjusted[idx, 2] += reduction * 0.5
        else:
            share = adjusted[idx, [0, 2]] / redistribution
            adjusted[idx, 0] += reduction * share[0]
            adjusted[idx, 2] += reduction * share[1]
    adjusted = np.clip(adjusted, 1e-6, 1.0)
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    return adjusted


def _softmax_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _apply_temperature_scaling(probas: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        temperature = 1.0
    logits = np.log(np.clip(probas, 1e-12, 1.0)) / temperature
    return _softmax_from_logits(logits)


def _fit_temperature_per_segment(
    probas: np.ndarray,
    y: np.ndarray,
    segments: np.ndarray,
    grid: Iterable[float] = np.linspace(0.6, 1.6, 11),
) -> Dict[str, float]:
    temps: Dict[str, float] = {}
    for seg in SEGMENT_NAMES:
        mask = segments == seg
        if not mask.any():
            temps[seg] = 1.0
            continue
        best_temp = 1.0
        best_ll = np.inf
        for temp in grid:
            calibrated = _apply_temperature_scaling(probas[mask], float(temp))
            ll = log_loss(y[mask], calibrated, labels=[0, 1, 2])
            if ll < best_ll:
                best_ll = ll
                best_temp = float(temp)
        temps[seg] = best_temp
    temps["global"] = 1.0
    return temps

def _apply_segmented_calibration(
    probas: np.ndarray,
    segments: np.ndarray,
    calibrators: Dict[str, IsotonicCalibrators | None],
    shrink: Dict[str, float] | None = None,
) -> np.ndarray:
    calibrated = probas.copy()
    applied_mask = np.zeros(len(probas), dtype=bool)
    for seg in SEGMENT_NAMES:
        mask = segments == seg
        calibrator = calibrators.get(seg)
        if calibrator is None or not mask.any():
            continue
        calibrated[mask] = apply_isotonic(calibrated[mask], calibrator)
        applied_mask |= mask
    global_cal = calibrators.get("global")
    if global_cal is not None:
        remaining = ~applied_mask
        if remaining.any():
            calibrated[remaining] = apply_isotonic(calibrated[remaining], global_cal)
        calibrated = apply_isotonic(calibrated, global_cal)
    if shrink:
        for seg in SEGMENT_NAMES:
            alpha = shrink.get(seg, 0.0)
            if alpha <= 0.0:
                continue
            mask = segments == seg
            if not mask.any():
                continue
            calibrated[mask] = (1.0 - alpha) * calibrated[mask] + alpha / 3.0
        global_alpha = shrink.get("global", 0.0)
        if global_alpha > 0.0:
            calibrated = (1.0 - global_alpha) * calibrated + global_alpha / 3.0
    calibrated = np.clip(calibrated, 1e-6, 1.0)
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    return calibrated


def _rolling_window_splits(n_samples: int, train_window: int = TRAIN_WINDOW, valid_window: int = VALID_WINDOW) -> List[RollingSplit]:
    splits: List[RollingSplit] = []
    if n_samples < train_window + MIN_VALID_MATCHES:
        raise ValueError("Insufficient samples to build rolling validation folds.")
    fold_id = 1
    start = train_window
    while start < n_samples:
        end = min(start + valid_window, n_samples)
        valid_length = end - start
        if valid_length < MIN_VALID_MATCHES:
            break
        train_start = max(0, start - train_window)
        train_idx = np.arange(train_start, start)
        valid_idx = np.arange(start, end)
        splits.append(RollingSplit(fold_id=fold_id, train_idx=train_idx, valid_idx=valid_idx))
        fold_id += 1
        start += valid_window
    if not splits:
        raise ValueError("Unable to construct any rolling validation windows.")
    return splits


def _prepare_dataset(team_dataset_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    fixtures = pd.read_csv(team_dataset_path, parse_dates=["match_date"])
    metadata_path = team_dataset_path.with_name("epl_unified_feature_columns.json")
    if metadata_path.exists():
        feature_columns = json.loads(metadata_path.read_text())
    else:
        inferred = [
            col
            for col in fixtures.columns
            if col.startswith(("home_feat_", "away_feat_", "diff_feat_"))
            or col in {"rest_days_home", "rest_days_away", "rest_days_diff"}
        ]
        if not inferred:
            raise ValueError("Unable to infer feature columns from unified dataset.")
        feature_columns = inferred
        metadata_path.write_text(json.dumps(feature_columns))
    if not feature_columns:
        raise ValueError("No Feature columns detected in unified dataset.")
    fixtures.sort_values(["match_date", "match_id", "home_team"], inplace=True)
    fixtures.reset_index(drop=True, inplace=True)
    return fixtures, feature_columns


def run_pipeline(team_dataset_path: Path, outputs_dir: Path) -> Dict[str, pd.DataFrame]:
    fixtures, feature_columns = _prepare_dataset(team_dataset_path)

    feature_set = [
        col
        for col in feature_columns
        if not col.startswith("book_")
        and not any(pattern in col for pattern in EXCLUDED_FEATURE_PATTERNS)
    ]
    X_all = fixtures[feature_set].fillna(0.0).astype(float)
    y_home_xg = fixtures["home_xg"].astype(float).fillna(0.0)
    y_away_xg = fixtures["away_xg"].astype(float).fillna(0.0)
    y_outcome_raw = fixtures["outcome"].fillna("draw")
    fixtures["label"] = y_outcome_raw
    y_outcome = y_outcome_raw.map(RESULT_TO_INT).astype(int)

    train_mask = fixtures["home_goals"].notna() & fixtures["away_goals"].notna()
    if train_mask.sum() < TRAIN_WINDOW + MIN_VALID_MATCHES:
        raise ValueError("Not enough played fixtures to run the redesigned pipeline.")

    train_fixtures = fixtures.loc[train_mask].copy()
    train_fixtures.sort_values(["match_date", "match_id"], inplace=True)
    train_fixtures["orig_index"] = train_fixtures.index.to_numpy()
    train_fixtures.reset_index(drop=True, inplace=True)

    X_train_full = X_all.loc[train_fixtures["orig_index"]].reset_index(drop=True)
    y_home_train = y_home_xg.loc[train_fixtures["orig_index"]].reset_index(drop=True)
    y_away_train = y_away_xg.loc[train_fixtures["orig_index"]].reset_index(drop=True)
    y_outcome_train = y_outcome.loc[train_fixtures["orig_index"]].reset_index(drop=True)

    selected_features = _prune_correlated_features(X_train_full, threshold=0.95)
    X_all = X_all[selected_features]
    X_train_full = X_train_full[selected_features]

    lambda_feature_cols = _select_columns_by_keywords(selected_features, LAMBDA_KEYWORDS)
    variance_feature_cols = _select_columns_by_keywords(selected_features, VARIANCE_KEYWORDS)
    X_lambda_all = X_all[lambda_feature_cols]
    X_lambda_train_full = X_train_full[lambda_feature_cols]
    X_variance_all = X_all[variance_feature_cols]
    X_variance_train_full = X_train_full[variance_feature_cols]

    splits = _rolling_window_splits(len(train_fixtures))

    oof_poisson = np.zeros((len(train_fixtures), 3))
    lambda_home_oof = np.zeros(len(train_fixtures))
    lambda_away_oof = np.zeros(len(train_fixtures))
    oof_gradient = np.zeros_like(oof_poisson)
    oof_form = np.zeros_like(oof_poisson)
    valid_mask = np.zeros(len(train_fixtures), dtype=bool)
    split_infos: List[RollingSplitInfo] = []

    for split in splits:
        train_idx = split.train_idx
        valid_idx = split.valid_idx

        X_train = X_train_full.iloc[train_idx]
        X_valid = X_train_full.iloc[valid_idx]
        X_train_lambda = X_lambda_train_full.iloc[train_idx]
        X_valid_lambda = X_lambda_train_full.iloc[valid_idx]
        y_home_tr = y_home_train.iloc[train_idx]
        y_away_tr = y_away_train.iloc[train_idx]
        y_outcome_tr = y_outcome_train.iloc[train_idx]

        home_reg = HistGradientBoostingRegressor(**REGRESSOR_PARAMS)
        away_reg = HistGradientBoostingRegressor(**REGRESSOR_PARAMS)
        home_reg.fit(X_train_lambda, y_home_tr)
        away_reg.fit(X_train_lambda, y_away_tr)

        home_lambda_train = np.clip(home_reg.predict(X_train_lambda), 0.05, 5.0)
        away_lambda_train = np.clip(away_reg.predict(X_train_lambda), 0.05, 5.0)
        home_lambda_valid = np.clip(home_reg.predict(X_valid_lambda), 0.05, 5.0)
        away_lambda_valid = np.clip(away_reg.predict(X_valid_lambda), 0.05, 5.0)

        lambda_home_oof[valid_idx] = home_lambda_valid
        lambda_away_oof[valid_idx] = away_lambda_valid

        poisson_valid = _poisson_match_probs(home_lambda_valid, away_lambda_valid)

        X_train_aug = _augment_with_xg(X_train, home_lambda_train, away_lambda_train)
        X_valid_aug = _augment_with_xg(X_valid, home_lambda_valid, away_lambda_valid)
        gradient_clf = HistGradientBoostingClassifier(**GRADIENT_CLASSIFIER_PARAMS)
        gradient_clf.fit(X_train_aug, y_outcome_tr)
        gradient_valid = gradient_clf.predict_proba(X_valid_aug)

        form_features = _select_form_features(X_train.columns)
        X_train_form = X_train[form_features]
        X_valid_form = X_valid[form_features]
        form_scaler = StandardScaler()
        X_train_form_scaled = form_scaler.fit_transform(X_train_form)
        X_valid_form_scaled = form_scaler.transform(X_valid_form)
        form_model = LogisticRegression(**FORM_LOGISTIC_PARAMS)
        form_model.fit(X_train_form_scaled, y_outcome_tr)
        form_valid = form_model.predict_proba(X_valid_form_scaled)

        oof_poisson[valid_idx] = poisson_valid
        oof_gradient[valid_idx] = gradient_valid
        oof_form[valid_idx] = form_valid
        valid_mask[valid_idx] = True

        valid_slice = train_fixtures.iloc[valid_idx]
        split_infos.append(
            RollingSplitInfo(
                fold_id=split.fold_id,
                train_idx=train_idx,
                valid_idx=valid_idx,
                train_matches=len(train_idx),
                valid_matches=len(valid_idx),
                valid_start=valid_slice["match_date"].iloc[0],
                valid_end=valid_slice["match_date"].iloc[-1],
            )
        )

    if not valid_mask.any():
        raise ValueError("No validation predictions generated; rolling splits may be misconfigured.")

    stack_mask = valid_mask
    rho = _fit_dc_rho(
        lambda_home_oof[stack_mask],
        lambda_away_oof[stack_mask],
        y_home_train.iloc[stack_mask].to_numpy(),
        y_away_train.iloc[stack_mask].to_numpy(),
    )
    oof_poisson[stack_mask] = _poisson_match_probs(
        lambda_home_oof[stack_mask], lambda_away_oof[stack_mask], rho
    )
    stack_features = np.hstack(
        [
            oof_poisson[stack_mask],
            oof_gradient[stack_mask],
            oof_form[stack_mask],
        ]
    )
    y_stack = y_outcome_train.iloc[stack_mask]

    meta_model = LogisticRegression(**META_STACK_PARAMS)
    meta_model.fit(stack_features, y_stack)

    oof_meta = np.zeros_like(oof_poisson)
    oof_meta[stack_mask] = meta_model.predict_proba(stack_features)
    segments_oof = _assign_segments(oof_meta[stack_mask])
    calibrators = _fit_segmented_calibrators(oof_meta[stack_mask], y_stack.to_numpy(), segments_oof)
    calibrated_preview = _apply_segmented_calibration(oof_meta[stack_mask], segments_oof, calibrators)
    shrink_factors = _compute_shrink_factors(calibrated_preview, y_stack.to_numpy(), segments_oof)
    for key, min_val in MIN_SHRINK.items():
        shrink_factors[key] = max(shrink_factors.get(key, 0.0), min_val)
    oof_calibrated = np.zeros_like(oof_meta)
    oof_calibrated[stack_mask] = _apply_segmented_calibration(
        oof_meta[stack_mask], segments_oof, calibrators, shrink_factors
    )
    variance_target = ((train_fixtures["home_goals"] - train_fixtures["away_goals"]) ** 2).astype(float)
    variance_model = HistGradientBoostingRegressor(**VARIANCE_PARAMS)
    variance_model.fit(X_variance_train_full, variance_target)
    variance_train_pred = variance_model.predict(X_variance_train_full)
    variance_all_pred = variance_model.predict(X_variance_all)

    oof_final = np.zeros_like(oof_calibrated)
    segment_temperatures = {seg: 1.0 for seg in list(SEGMENT_NAMES) + ["global"]}
    if stack_mask.any():
        combined = 0.8 * oof_calibrated[stack_mask] + 0.2 * oof_poisson[stack_mask]
        combined = combined / combined.sum(axis=1, keepdims=True)
        guarded = _apply_draw_guardrails(
            combined,
            lambda_home_oof[stack_mask],
            lambda_away_oof[stack_mask],
            variance_train_pred[stack_mask],
        )
        guarded[:, 1] *= DRAW_DAMPING
        guarded = guarded / guarded.sum(axis=1, keepdims=True)
        guarded = (1.0 - FINAL_SHRINK_DEFAULT) * guarded + FINAL_SHRINK_DEFAULT / 3.0
        guarded = guarded / guarded.sum(axis=1, keepdims=True)
        segments_final_oof = _assign_segments(guarded)
        segment_temperatures.update(
            _fit_temperature_per_segment(guarded, y_stack.to_numpy(), segments_final_oof)
        )
        for seg, temp in segment_temperatures.items():
            mask = segments_final_oof == seg
            if seg == "global" or not mask.any():
                continue
            guarded[mask] = _apply_temperature_scaling(guarded[mask], temp)
        guarded = _apply_temperature_scaling(guarded, segment_temperatures.get("global", 1.0))
        guarded = guarded / guarded.sum(axis=1, keepdims=True)
        oof_final[stack_mask] = guarded

    final_calibrators = {seg: None for seg in list(SEGMENT_NAMES) + ["global"]}
    final_shrink = {seg: 0.0 for seg in list(SEGMENT_NAMES) + ["global"]}

    validation_rows: List[Dict[str, object]] = []
    for info in split_infos:
        idx = info.valid_idx
        y_true_fold = y_outcome_train.iloc[idx].to_numpy()
        probs_fold = oof_final[idx]
        metrics = evaluate_metrics(y_true_fold, probs_fold)
        metrics.update(
            {
                "fold": info.label,
                "train_matches": info.train_matches,
                "valid_matches": info.valid_matches,
                "valid_start": info.valid_start.strftime("%Y-%m-%d"),
                "valid_end": info.valid_end.strftime("%Y-%m-%d"),
            }
        )
        validation_rows.append(metrics)

    validation_metrics = pd.DataFrame(validation_rows)

    home_reg_full = HistGradientBoostingRegressor(**REGRESSOR_PARAMS)
    away_reg_full = HistGradientBoostingRegressor(**REGRESSOR_PARAMS)
    home_reg_full.fit(X_lambda_train_full, y_home_train)
    away_reg_full.fit(X_lambda_train_full, y_away_train)

    home_lambda_all = np.clip(home_reg_full.predict(X_lambda_all), 0.05, 5.0)
    away_lambda_all = np.clip(away_reg_full.predict(X_lambda_all), 0.05, 5.0)
    poisson_all = _poisson_match_probs(home_lambda_all, away_lambda_all, rho)

    home_lambda_train_full = np.clip(home_reg_full.predict(X_lambda_train_full), 0.05, 5.0)
    away_lambda_train_full = np.clip(away_reg_full.predict(X_lambda_train_full), 0.05, 5.0)
    X_train_aug_full = _augment_with_xg(X_train_full, home_lambda_train_full, away_lambda_train_full)
    X_all_aug = _augment_with_xg(X_all, home_lambda_all, away_lambda_all)
    gradient_model_full = HistGradientBoostingClassifier(**GRADIENT_CLASSIFIER_PARAMS)
    gradient_model_full.fit(X_train_aug_full, y_outcome_train)
    gradient_all = gradient_model_full.predict_proba(X_all_aug)

    form_features_full = _select_form_features(X_train_full.columns)
    scaler_full = StandardScaler()
    X_train_form_full = scaler_full.fit_transform(X_train_full[form_features_full])
    form_model_full = LogisticRegression(**FORM_LOGISTIC_PARAMS)
    form_model_full.fit(X_train_form_full, y_outcome_train)
    X_all_form_scaled = scaler_full.transform(X_all[form_features_full])
    form_all = form_model_full.predict_proba(X_all_form_scaled)

    stack_all = np.hstack([poisson_all, gradient_all, form_all])
    meta_all = meta_model.predict_proba(stack_all)
    segments_all = _assign_segments(meta_all)
    calibrated_probs = _apply_segmented_calibration(meta_all, segments_all, calibrators, shrink_factors)
    blended_probs = 0.8 * calibrated_probs + 0.2 * poisson_all
    blended_probs = blended_probs / blended_probs.sum(axis=1, keepdims=True)
    final_probs = _apply_draw_guardrails(blended_probs, home_lambda_all, away_lambda_all, variance_all_pred)
    final_probs[:, 1] *= DRAW_DAMPING
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
    final_probs = (1.0 - FINAL_SHRINK_DEFAULT) * final_probs + FINAL_SHRINK_DEFAULT / 3.0
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
    segments_final_all = _assign_segments(final_probs)
    for seg, temp in segment_temperatures.items():
        if seg == "global":
            continue
        mask = segments_final_all == seg
        if not mask.any():
            continue
        final_probs[mask] = _apply_temperature_scaling(final_probs[mask], temp)
    final_probs = _apply_temperature_scaling(final_probs, segment_temperatures.get("global", 1.0))
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
    confidence = final_probs.max(axis=1) - final_probs.min(axis=1)
    recommendations, top_conf, second_conf, predicted_class = _recommendation_flags(final_probs)

    odds_cols = ["home_odds_win", "home_odds_draw", "home_odds_loss"]
    if set(odds_cols).issubset(fixtures.columns):
        implied = _implied_probs(fixtures[odds_cols])
    else:
        implied = pd.DataFrame(np.nan, index=fixtures.index, columns=["book_prob_home", "book_prob_draw", "book_prob_away"])

    final_predictions = fixtures.copy()
    final_predictions = pd.concat([final_predictions.reset_index(drop=True), implied.reset_index(drop=True)], axis=1)
    final_predictions["pred_home_xg"] = home_lambda_all
    final_predictions["pred_away_xg"] = away_lambda_all
    final_predictions["pred_xg_total"] = home_lambda_all + away_lambda_all
    final_predictions["pred_xg_diff"] = home_lambda_all - away_lambda_all
    final_predictions["pred_goal_variance"] = variance_all_pred
    final_predictions[["p_poisson_home", "p_poisson_draw", "p_poisson_away"]] = poisson_all
    final_predictions[["p_gradient_home", "p_gradient_draw", "p_gradient_away"]] = gradient_all
    final_predictions[["p_form_home", "p_form_draw", "p_form_away"]] = form_all
    final_predictions[["p_meta_home", "p_meta_draw", "p_meta_away"]] = meta_all
    final_predictions[["p_calibrated_home", "p_calibrated_draw", "p_calibrated_away"]] = calibrated_probs
    final_predictions[["final_p_home", "final_p_draw", "final_p_away"]] = final_probs
    final_predictions["confidence"] = confidence
    final_predictions["confidence_top"] = top_conf
    final_predictions["confidence_second"] = second_conf
    final_predictions["prediction"] = predicted_class
    final_predictions["prediction"] = final_predictions["prediction"].map(INT_TO_RESULT)
    final_predictions["recommended"] = recommendations
    final_predictions["recommended_pick"] = np.where(recommendations, final_predictions["prediction"], "none")
    final_predictions["delta_lambda"] = np.abs(final_predictions["pred_xg_diff"])
    final_predictions["total_lambda"] = final_predictions["pred_xg_total"]
    final_predictions["correct"] = np.where(
        final_predictions["label"].notna(),
        final_predictions["prediction"] == final_predictions["label"],
        np.nan,
    )

    played_mask = final_predictions["home_goals"].notna()
    played_idx = np.where(played_mask.to_numpy(dtype=bool))[0]
    rec_played_mask = recommendations & played_mask.to_numpy(dtype=bool)
    rec_idx = np.where(rec_played_mask)[0]
    oof_indices = np.where(valid_mask)[0]
    if oof_indices.size:
        y_oof = y_outcome_train.iloc[oof_indices].to_numpy()
        oof_probs = oof_final[oof_indices]
        oof_metrics = evaluate_metrics(y_oof, oof_probs)
        oof_recs, _, _, oof_pred = _recommendation_flags(oof_probs)
        if oof_recs.any():
            oof_correct = (oof_pred[oof_recs] == y_oof[oof_recs])
            recommendation_accuracy = float(oof_correct.mean())
        else:
            recommendation_accuracy = float("nan")
    else:
        oof_metrics = {"accuracy": np.nan, "brier": np.nan, "logloss": np.nan, "ece": np.nan}
        recommendation_accuracy = float("nan")

    if played_idx.size:
        y_true_all = final_predictions.iloc[played_idx]["label"].map(RESULT_TO_INT).to_numpy()
        probs_played = final_probs[played_idx]
        overall_metrics = evaluate_metrics(y_true_all, probs_played)
    else:
        overall_metrics = {"accuracy": np.nan, "brier": np.nan, "logloss": np.nan, "ece": np.nan}

    season_rows: List[Dict[str, object]] = []
    for season, group in final_predictions.loc[played_mask].groupby("season"):
        y_true = group["label"].map(RESULT_TO_INT).to_numpy()
        probs = group[["final_p_home", "final_p_draw", "final_p_away"]].to_numpy()
        metrics = evaluate_metrics(y_true, probs)
        metrics.update({"season": season})
        season_rows.append(metrics)
    season_metrics = pd.DataFrame(season_rows).sort_values("season")
    if not season_metrics.empty:
        season_metrics.rename(
            columns={
                "accuracy": "model_accuracy",
                "brier": "model_brier_score",
                "logloss": "model_logloss",
                "ece": "model_ece",
            },
            inplace=True,
        )
        season_metrics["accuracy_change_from_previous"] = season_metrics["model_accuracy"].diff()

    if played_idx.size:
        prob_means_played = final_probs[played_idx].mean(axis=0)
    else:
        prob_means_played = np.array([np.nan, np.nan, np.nan])

    summary = pd.DataFrame(
        [
            {"metric": "validation_accuracy_mean", "value": float(validation_metrics["accuracy"].mean()) if not validation_metrics.empty else np.nan},
            {"metric": "validation_logloss_mean", "value": float(validation_metrics["logloss"].mean()) if not validation_metrics.empty else np.nan},
            {"metric": "validation_brier_mean", "value": float(validation_metrics["brier"].mean()) if not validation_metrics.empty else np.nan},
            {"metric": "validation_ece_mean", "value": float(validation_metrics["ece"].mean()) if not validation_metrics.empty else np.nan},
            {"metric": "oof_accuracy", "value": oof_metrics["accuracy"]},
            {"metric": "oof_logloss", "value": oof_metrics["logloss"]},
            {"metric": "oof_brier", "value": oof_metrics["brier"]},
            {"metric": "oof_ece", "value": oof_metrics["ece"]},
            {"metric": "dc_rho", "value": rho},
            {"metric": "recommendation_count", "value": int(recommendations.sum())},
            {"metric": "recommendation_accuracy", "value": recommendation_accuracy},
            {"metric": "prob_home_mean", "value": prob_means_played[0]},
            {"metric": "prob_draw_mean", "value": prob_means_played[1]},
            {"metric": "prob_away_mean", "value": prob_means_played[2]},
        ]
    )

    future = final_predictions[~played_mask].copy()
    forecast_clean = (
        future[
            [
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
                "pred_home_xg",
                "pred_away_xg",
                "pred_xg_total",
                "pred_xg_diff",
                "pred_goal_variance",
                "book_prob_home",
                "book_prob_draw",
                "book_prob_away",
                "confidence",
            ]
        ]
        .drop_duplicates(subset=["match_id"])
        .sort_values("match_date")
    )

    diagnostics = compare_model_vs_market(
        forecast_clean.rename(
            columns={
                "final_p_home": "p_home_win",
                "final_p_draw": "p_draw",
                "final_p_away": "p_away_win",
            }
        )
    )

    draw_rate_vs_gap = pd.DataFrame()
    if played_idx.size:
        played_final = final_predictions.iloc[played_idx].copy()
        played_final["pred_draw_prob"] = final_probs[played_idx, 1]
        played_final["actual_draw"] = (played_final["label"] == "draw").astype(float)
        delta_edges = np.linspace(0.0, 2.0, 11)
        total_edges = [0.0, 2.0, 3.0, 4.5, 10.0]
        played_final["delta_bin"] = pd.cut(
            played_final["delta_lambda"], bins=delta_edges, include_lowest=True, right=False
        )
        played_final["total_bin"] = pd.cut(
            played_final["total_lambda"], bins=total_edges, include_lowest=True, right=False
        )
        draw_rate_vs_gap = (
            played_final.groupby(["delta_bin", "total_bin"], observed=True)
            .agg(
                predicted_draw=("pred_draw_prob", "mean"),
                observed_draw=("actual_draw", "mean"),
                fixtures=("actual_draw", "count"),
            )
            .reset_index()
        )

    outputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = outputs_dir / "model_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    calibrated_export = final_predictions.copy()
    calibrated_export[["final_p_home", "final_p_draw", "final_p_away"]] = calibrated_probs
    calibrated_export.to_csv(outputs_dir / "final_predictions_calibrated.csv", index=False)

    final_predictions.to_csv(outputs_dir / "final_predictions.csv", index=False)
    validation_metrics.to_csv(outputs_dir / "validation_metrics.csv", index=False)
    season_metrics.to_csv(outputs_dir / "season_metrics.csv", index=False)
    summary.to_csv(outputs_dir / "summary.csv", index=False)
    forecast_clean.to_csv(outputs_dir / "forecast_clean.csv", index=False)
    final_predictions.to_csv(outputs_dir / "forecast_raw.csv", index=False)
    diagnostics.to_csv(outputs_dir / "diagnostics_market_gap.csv", index=False)
    if not draw_rate_vs_gap.empty:
        draw_rate_vs_gap.to_csv(outputs_dir / "draw_rate_vs_gap.csv", index=False)

    joblib.dump(home_reg_full, artifacts_dir / "home_xg_regressor.pkl")
    joblib.dump(away_reg_full, artifacts_dir / "away_xg_regressor.pkl")
    joblib.dump(gradient_model_full, artifacts_dir / "gradient_classifier.pkl")
    joblib.dump(form_model_full, artifacts_dir / "form_logistic.pkl")
    joblib.dump(scaler_full, artifacts_dir / "form_scaler.pkl")
    (artifacts_dir / "form_features.json").write_text(json.dumps(form_features_full))
    joblib.dump(meta_model, artifacts_dir / "meta_stacker.pkl")
    joblib.dump(calibrators, artifacts_dir / "segmented_calibrators.pkl")
    joblib.dump(variance_model, artifacts_dir / "variance_regressor.pkl")
    (artifacts_dir / "shrink_factors.json").write_text(json.dumps(shrink_factors))
    (artifacts_dir / "selected_features.json").write_text(json.dumps(feature_set))

    readme = outputs_dir / "README_forecast.md"
    readme.write_text(
        "# Fixture Forecast Outputs\n\n"
        "- `forecast_raw.csv`: model probabilities and diagnostics for all fixtures.\n"
        "- `forecast_clean.csv`: upcoming fixtures (home perspective) with calibrated probabilities.\n"
        "- `draw_rate_vs_gap.csv`: draw-rate calibration by expected goal gap and intensity.\n"
        "- `diagnostics_market_gap.csv`: bookmaker comparison (for analysis only).\n"
        "- `summary.csv`, `validation_metrics.csv`, `season_metrics.csv`: performance reports.\n"
    )

    return {
        "summary": summary,
        "validation_metrics": validation_metrics,
        "season_metrics": season_metrics,
        "predictions": final_predictions,
        "forecast": forecast_clean,
    }
