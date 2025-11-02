"""
High-level orchestration for downloading data, preparing features, running the
online model, and saving outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data import DatasetConfig
from .config_loader import FootballConfig, load_config
from .download import SeasonSpec, download_multiple
from .evaluation import RESULT_ORDER, evaluate_predictions
from .features import build_dataset, save_dataset
from .goal_model import (
    poisson_outcome_probabilities,
    train_goal_expectations,
)
from .pipeline import PipelineConfig, run_pipeline
from .unified_dataset import augment_unified_features


@dataclass
class WorkflowPaths:
    base_dir: Path
    raw_dir: Path
    processed_dir: Path
    outputs_dir: Path

    @classmethod
    def default(cls, root: Path) -> "WorkflowPaths":
        return cls(
            base_dir=root,
            raw_dir=root / "data" / "raw",
            processed_dir=root / "data" / "processed",
            outputs_dir=root / "outputs",
        )


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        pass


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass


def seasons_from_config(config: FootballConfig) -> List[SeasonSpec]:
    return [
        SeasonSpec(label=season["label"], code=season["code"]) for season in config.seasons
    ]


def prepare_dataset(paths: WorkflowPaths, seasons: List[SeasonSpec]) -> Path:
    download_multiple(seasons, paths.raw_dir)
    raw_files = {
        season.label: paths.raw_dir / f"epl_{season.code}.csv"
        for season in seasons
    }
    dataset = build_dataset(raw_files)
    output_path = paths.processed_dir / "epl_rolling_features.csv"
    save_dataset(dataset, output_path)
    return output_path


def run_full_workflow(
    dataset_path: Path,
    paths: WorkflowPaths,
    feature_columns: List[str],
    config: FootballConfig,
    baseline_prob_columns: Dict[str, str] | None = None,
) -> Dict[str, pd.DataFrame]:
    config = PipelineConfig(
        dataset_path=dataset_path,
        dataset_config=DatasetConfig(),
        feature_columns=feature_columns,
        seasons=[season["label"] for season in config.seasons],
        learning_rate=config.learning_rate,
        warmup_matches=config.warmup_matches,
        baseline_prob_columns=baseline_prob_columns,
    )
    outputs = run_pipeline(config)

    _ensure_directory(paths.outputs_dir)
    _safe_to_csv(outputs["predictions"], paths.outputs_dir / "predictions.csv")
    _safe_to_csv(outputs["season_metrics"], paths.outputs_dir / "season_metrics.csv")
    _safe_to_csv(outputs["summary"], paths.outputs_dir / "summary.csv")
    return outputs


def collect_feature_columns(dataset_path: Path) -> List[str]:
    df = pd.read_csv(dataset_path, nrows=1)

    allowed_prefixes = (
        "feat_",
        "team_xg_rolling_",
        "opponent_xg_rolling_",
        "xg_diff_rolling_",
    )
    allowed_names = {
        "attack_strength",
        "defence_weakness",
        "attack_vs_defence",
        "xg_form_mismatch",
        "team_match_number",
        "rest_days",
        "opponent_rest_days",
        "rest_days_diff",
        "team_points_before",
        "team_points_avg_5",
        "team_points_home_avg_5",
        "team_points_away_avg_5",
        "opponent_points_before",
        "opponent_points_avg_5",
        "opponent_points_home_avg_5",
        "opponent_points_away_avg_5",
        "team_strength_index",
        "opponent_strength_index",
        "strength_index_diff",
        "points_avg_5_diff",
        "team_xg_diff_std_5",
        "opponent_xg_diff_std_5",
        "matchup_xg_diff_avg_3",
        "team_expected_points_market",
        "opponent_expected_points_market",
        "expected_points_edge",
        "market_confidence_gap",
        "opponent_book_prob_win",
        "opponent_book_prob_draw",
        "opponent_book_prob_loss",
        "team_points_trend_5",
        "team_points_trend_8",
        "team_xg_trend_5",
        "team_xg_trend_8",
        "opponent_points_trend_5",
        "opponent_points_trend_8",
        "opponent_xg_trend_5",
        "opponent_xg_trend_8",
        "points_trend_diff_5",
        "points_trend_diff_8",
        "xg_trend_diff_5",
        "xg_trend_diff_8",
        "team_xg_conversion_5",
        "team_xg_conversion_8",
        "team_xg_overperformance_5",
        "team_xg_overperformance_8",
        "opponent_xg_conversion_5",
        "opponent_xg_conversion_8",
        "opponent_xg_overperformance_5",
        "opponent_xg_overperformance_8",
        "xg_conversion_ratio_diff_5",
        "xg_conversion_ratio_diff_8",
        "xg_overperformance_diff_5",
        "xg_overperformance_diff_8",
        "team_elo_pre",
        "opponent_elo_pre",
        "elo_diff",
        "elo_expected_score",
        "elo_edge",
        "eg_lambda_team",
        "eg_lambda_opponent",
        "eg_lambda_diff",
        "eg_prob_win",
        "eg_prob_draw",
        "eg_prob_loss",
        "eg_prob_win_edge",
        "eg_prob_draw_edge",
        "eg_prob_loss_edge",
    }

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    feature_cols = []
    for col in numeric_cols:
        if col.startswith(allowed_prefixes) or col in allowed_names:
            feature_cols.append(col)

    ordered = list(dict.fromkeys(feature_cols))
    return ordered


def default_config() -> FootballConfig:
    return load_config()


def _accuracy(predicted: pd.Series, actual: pd.Series) -> float:
    mask = predicted.notna() & actual.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((predicted[mask] == actual[mask]).mean())


def _brier_score(probabilities: pd.DataFrame, actual: pd.Series) -> float:
    mask = actual.notna() & probabilities.notna().all(axis=1)
    if mask.sum() == 0:
        return float("nan")
    probs = probabilities.loc[mask].to_numpy(dtype=float)
    indices = actual.loc[mask].map(lambda cls: RESULT_ORDER.index(cls)).to_numpy()
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(indices)), indices] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def build_ensemble_outputs(
    online_predictions: pd.DataFrame,
    offline_predictions: pd.DataFrame,
    paths: WorkflowPaths,
    weights: Dict[str, float] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Combine online model probabilities, offline booster probabilities, and bookmaker
    implied probabilities into an ensemble forecast.
    """
    if weights is None:
        weights = {"online": 0.25, "offline": 0.5, "goals": 0.15, "book": 0.1}

    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    weights = {k: v / total_weight for k, v in weights.items()}

    offline_probs = offline_predictions[
        ["match_id", "team"] + [f"pred_prob_{cls}" for cls in RESULT_ORDER]
    ].rename(columns={f"pred_prob_{cls}": f"offline_prob_{cls}" for cls in RESULT_ORDER})

    merged = online_predictions.merge(offline_probs, on=["match_id", "team"], how="left")

    for cls in RESULT_ORDER:
        goal_col = f"eg_prob_{cls}"
        if goal_col not in merged.columns:
            merged[goal_col] = 1.0 / len(RESULT_ORDER)
        merged[goal_col] = merged[goal_col].fillna(1.0 / len(RESULT_ORDER))

    for cls in RESULT_ORDER:
        model_col = f"model_prob_{cls}"
        if model_col not in merged.columns:
            merged[model_col] = 1.0 / len(RESULT_ORDER)
        merged[model_col] = merged[model_col].fillna(1.0 / len(RESULT_ORDER))

    # If offline probabilities missing, default to online probabilities
    for cls in RESULT_ORDER:
        merged[f"offline_prob_{cls}"] = merged[f"offline_prob_{cls}"].fillna(
            merged[f"model_prob_{cls}"]
        )

    # Ensure bookmaker probabilities exist
    for cls in RESULT_ORDER:
        book_col = f"book_prob_{cls}"
        if book_col not in merged.columns:
            merged[book_col] = 1.0 / len(RESULT_ORDER)
        merged[book_col] = merged[book_col].fillna(1.0 / len(RESULT_ORDER))

    feature_cols = []
    for cls in RESULT_ORDER:
        feature_cols.append(f"model_prob_{cls}")
        feature_cols.append(f"offline_prob_{cls}")
        feature_cols.append(f"book_prob_{cls}")
        feature_cols.append(f"eg_prob_{cls}")

    train_mask = merged["actual"].notna()
    meta_probs = None
    if train_mask.sum() >= len(RESULT_ORDER) * 3:
        try:
            from sklearn.linear_model import LogisticRegression

            X_train = merged.loc[train_mask, feature_cols].to_numpy(dtype=float)
            y_train = merged.loc[train_mask, "actual"].map(lambda cls: RESULT_ORDER.index(cls)).to_numpy()
            meta_model = LogisticRegression(max_iter=200, solver="lbfgs")
            meta_model.fit(X_train, y_train)
            X_all = merged[feature_cols].to_numpy(dtype=float)
            meta_probs = meta_model.predict_proba(X_all)
        except Exception:
            meta_probs = None

    if meta_probs is not None:
        ensemble_matrix = meta_probs
    else:
        ensemble_probs = []
        for cls in RESULT_ORDER:
            prob = (
                weights["online"] * merged[f"model_prob_{cls}"]
                + weights["offline"] * merged[f"offline_prob_{cls}"]
                + weights.get("goals", 0.0) * merged[f"eg_prob_{cls}"]
                + weights["book"] * merged[f"book_prob_{cls}"]
            )
            ensemble_probs.append(prob)
        ensemble_matrix = np.column_stack(ensemble_probs)
        ensemble_matrix = np.clip(ensemble_matrix, 1e-8, 1.0)
        ensemble_matrix = ensemble_matrix / ensemble_matrix.sum(axis=1, keepdims=True)

    for idx, cls in enumerate(RESULT_ORDER):
        merged[f"ensemble_prob_{cls}"] = ensemble_matrix[:, idx]

    ensemble_indices = ensemble_matrix.argmax(axis=1)
    merged["ensemble_prediction"] = [RESULT_ORDER[idx] for idx in ensemble_indices]
    merged["ensemble_correct"] = np.where(
        merged["actual"].notna(),
        merged["ensemble_prediction"] == merged["actual"],
        np.nan,
    )

    # Compute metrics
    ensemble_accuracy = _accuracy(merged["ensemble_prediction"], merged["actual"])
    ensemble_brier = _brier_score(
        merged[[f"ensemble_prob_{cls}" for cls in RESULT_ORDER]],
        merged["actual"],
    )

    confidence = np.max(ensemble_matrix, axis=1)
    hc_mask = merged["actual"].notna() & (confidence >= 0.6)
    ensemble_high_conf_accuracy = float(merged.loc[hc_mask, "ensemble_prediction"].eq(merged.loc[hc_mask, "actual"]).mean()) if hc_mask.any() else np.nan
    ensemble_high_conf_coverage = float(hc_mask.mean())

    summary = pd.DataFrame(
        [
            {
                "metric": "ensemble_accuracy",
                "value": ensemble_accuracy,
            },
            {
                "metric": "ensemble_brier",
                "value": ensemble_brier,
            },
            {
                "metric": "ensemble_high_conf_accuracy",
                "value": ensemble_high_conf_accuracy,
            },
            {
                "metric": "ensemble_high_conf_coverage",
                "value": ensemble_high_conf_coverage,
            },
        ]
    )

    _ensure_directory(paths.outputs_dir)
    _safe_to_csv(merged, paths.outputs_dir / "ensemble_predictions.csv")

    return {
        "predictions": merged,
        "summary": summary,
    }


def build_online_offline_blend(
    online_predictions: pd.DataFrame,
    offline_predictions: pd.DataFrame,
    paths: WorkflowPaths,
    weights: Dict[str, float] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Blend only the online model and offline model probabilities (no bookmaker/xG).
    """
    if weights is None:
        weights = {"online": 0.65, "offline": 0.35}

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Blend weights must sum to a positive value.")
    weights = {k: v / total_weight for k, v in weights.items()}

    offline_probs = offline_predictions[
        ["match_id", "team"] + [f"pred_prob_{cls}" for cls in RESULT_ORDER]
    ].rename(columns={f"pred_prob_{cls}": f"offline_prob_{cls}" for cls in RESULT_ORDER})

    merged = online_predictions.merge(offline_probs, on=["match_id", "team"], how="left")

    offline_cols = [f"offline_prob_{cls}" for cls in RESULT_ORDER]
    offline_available = merged[offline_cols].notna().all(axis=1)

    for cls in RESULT_ORDER:
        merged[f"model_prob_{cls}"] = merged[f"model_prob_{cls}"].fillna(1.0 / len(RESULT_ORDER))
        merged[f"offline_prob_{cls}"] = merged[f"offline_prob_{cls}"].fillna(merged[f"model_prob_{cls}"])

    feature_cols = []
    for cls in RESULT_ORDER:
        feature_cols.append(f"model_prob_{cls}")
        feature_cols.append(f"offline_prob_{cls}")

    train_mask = merged["actual"].notna()
    meta_probs = None
    if train_mask.sum() >= len(RESULT_ORDER) * 3:
        try:
            from sklearn.linear_model import LogisticRegression

            X_train = merged.loc[train_mask, feature_cols].to_numpy(dtype=float)
            y_train = merged.loc[train_mask, "actual"].map(lambda cls: RESULT_ORDER.index(cls)).to_numpy()
            meta_model = LogisticRegression(max_iter=200, solver="lbfgs")
            meta_model.fit(X_train, y_train)
            X_all = merged[feature_cols].to_numpy(dtype=float)
            meta_probs = meta_model.predict_proba(X_all)
        except Exception:
            meta_probs = None

    if meta_probs is not None:
        blend_matrix = meta_probs
    else:
        blend_probs = []
        for cls in RESULT_ORDER:
            prob = (
                weights["online"] * merged[f"model_prob_{cls}"]
                + weights["offline"] * merged[f"offline_prob_{cls}"]
            )
            blend_probs.append(prob)
        blend_matrix = np.column_stack(blend_probs)
        blend_matrix = np.clip(blend_matrix, 1e-8, 1.0)
        blend_matrix = blend_matrix / blend_matrix.sum(axis=1, keepdims=True)

    if not offline_available.all():
        mask_missing = ~offline_available.to_numpy()
        replacement = merged.loc[~offline_available, [f"model_prob_{cls}" for cls in RESULT_ORDER]].to_numpy(dtype=float)
        blend_matrix[mask_missing] = replacement

    for idx, cls in enumerate(RESULT_ORDER):
        merged[f"final_prob_{cls}"] = blend_matrix[:, idx]

    blend_indices = blend_matrix.argmax(axis=1)
    merged["final_prediction"] = [RESULT_ORDER[idx] for idx in blend_indices]
    merged["final_correct"] = np.where(
        merged["actual"].notna(),
        merged["final_prediction"] == merged["actual"],
        np.nan,
    )

    final_accuracy = _accuracy(merged["final_prediction"], merged["actual"])
    final_brier = _brier_score(
        merged[[f"final_prob_{cls}" for cls in RESULT_ORDER]],
        merged["actual"],
    )

    summary = pd.DataFrame(
        [
            {"metric": "final_blend_accuracy", "value": final_accuracy},
            {"metric": "final_blend_brier", "value": final_brier},
        ]
    )

    _ensure_directory(paths.outputs_dir)
    _safe_to_csv(merged, paths.outputs_dir / "final_predictions.csv")
    _safe_to_csv(summary, paths.outputs_dir / "final_summary.csv")

    return {"predictions": merged, "summary": summary}


def train_offline_model(
    dataset_path: Path,
    paths: WorkflowPaths,
    config: FootballConfig,
) -> Dict[str, pd.DataFrame]:
    """
    Train an offline gradient boosting model using the processed dataset.
    Returns predictions, metrics, and summary data frames.
    """
    df = pd.read_csv(dataset_path, parse_dates=["match_date"])
    df = augment_unified_features(df)
    df = df[df["result_class"].notna()].copy()
    df = df[df["team_match_number"] > config.warmup_matches].copy()

    if df.empty:
        raise ValueError("No labelled records available for offline training.")

    df.sort_values(["match_date", "team"], inplace=True)

    base_feature_columns = collect_feature_columns(dataset_path)

    odds_columns = [col for col in ["odds_win", "odds_draw", "odds_loss"] if col in df.columns]
    prob_columns = [col for col in ["book_prob_win", "book_prob_draw", "book_prob_loss"] if col in df.columns]
    if not prob_columns and set(["odds_win", "odds_draw", "odds_loss"]).issubset(df.columns):
        odds_frame = df[["odds_win", "odds_draw", "odds_loss"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = 1.0 / odds_frame.replace(0, np.nan)
        implied_sum = implied.sum(axis=1)
        prob_df = implied.div(implied_sum, axis=0).fillna(0.0)
        prob_columns = ["book_prob_win", "book_prob_draw", "book_prob_loss"]
        df.loc[:, prob_columns] = prob_df.to_numpy()

    df["venue_is_home"] = (df["venue"] == "home").astype(int)

    label_series = df["result_class"].astype(str)

    train_size = int(len(df) * 0.8)
    if train_size == 0 or train_size == len(df):
        raise ValueError("Dataset too small for 80/20 split.")

    unique_classes = set(label_series.unique())
    while set(label_series.iloc[:train_size]) != unique_classes and train_size < len(df) - 1:
        train_size += 1
    while set(label_series.iloc[train_size:]) != unique_classes and train_size > 1:
        train_size -= 1
    if train_size == 0 or train_size == len(df):
        raise ValueError("Unable to create class-balanced chronological split.")

    X_base = df[base_feature_columns].fillna(0.0)
    y_team_all = df["team_goals"].astype(float)
    y_opp_all = df["opponent_goals"].astype(float)

    # Expected goal features using expanding time splits
    lambda_team_train, lambda_team_test, metrics_team = train_goal_expectations(
        X_base.iloc[:train_size], y_team_all.iloc[:train_size], X_base.iloc[train_size:]
    )
    lambda_opp_train, lambda_opp_test, metrics_opp = train_goal_expectations(
        X_base.iloc[:train_size], y_opp_all.iloc[:train_size], X_base.iloc[train_size:]
    )

    prob_train = poisson_outcome_probabilities(lambda_team_train, lambda_opp_train)
    prob_test = poisson_outcome_probabilities(lambda_team_test, lambda_opp_test)

    # Ensure columns exist before positional assignment
    df["eg_lambda_team"] = np.nan
    df["eg_lambda_opponent"] = np.nan
    df["eg_lambda_diff"] = np.nan
    df["eg_prob_win"] = np.nan
    df["eg_prob_draw"] = np.nan
    df["eg_prob_loss"] = np.nan

    df.iloc[:train_size, df.columns.get_loc("eg_lambda_team")] = lambda_team_train
    df.iloc[train_size:, df.columns.get_loc("eg_lambda_team")] = lambda_team_test
    df.iloc[:train_size, df.columns.get_loc("eg_lambda_opponent")] = lambda_opp_train
    df.iloc[train_size:, df.columns.get_loc("eg_lambda_opponent")] = lambda_opp_test

    df["eg_lambda_diff"] = df["eg_lambda_team"] - df["eg_lambda_opponent"]

    df.iloc[:train_size, df.columns.get_loc("eg_prob_win")] = prob_train[:, 0]
    df.iloc[:train_size, df.columns.get_loc("eg_prob_draw")] = prob_train[:, 1]
    df.iloc[:train_size, df.columns.get_loc("eg_prob_loss")] = prob_train[:, 2]
    df.iloc[train_size:, df.columns.get_loc("eg_prob_win")] = prob_test[:, 0]
    df.iloc[train_size:, df.columns.get_loc("eg_prob_draw")] = prob_test[:, 1]
    df.iloc[train_size:, df.columns.get_loc("eg_prob_loss")] = prob_test[:, 2]

    if {"book_prob_win", "book_prob_draw", "book_prob_loss"}.issubset(df.columns):
        df["eg_prob_win_edge"] = df["eg_prob_win"] - df["book_prob_win"]
        df["eg_prob_draw_edge"] = df["eg_prob_draw"] - df["book_prob_draw"]
        df["eg_prob_loss_edge"] = df["eg_prob_loss"] - df["book_prob_loss"]

    goal_feature_names = [
        "eg_lambda_team",
        "eg_lambda_opponent",
        "eg_lambda_diff",
        "eg_prob_win",
        "eg_prob_draw",
        "eg_prob_loss",
    ]

    edge_cols = [
        col
        for col in ["eg_prob_win_edge", "eg_prob_draw_edge", "eg_prob_loss_edge"]
        if col in df.columns
    ]
    goal_feature_names.extend(edge_cols)

    feature_columns = base_feature_columns.copy()
    for col in goal_feature_names:
        if col not in feature_columns:
            feature_columns.append(col)

    all_features = feature_columns + odds_columns + prob_columns + ["venue_is_home"]
    if not all_features:
        raise ValueError("No usable features found for offline training.")

    X = df[all_features].fillna(0.0)
    le = LabelEncoder().fit(label_series)
    y_encoded = le.transform(label_series)

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y_encoded[:train_size], y_encoded[train_size:]
    df_test = df.iloc[train_size:].copy()

    random_seed = int(config.raw.get("project", {}).get("random_seed", 21))

    base_models = {
        "hist": HistGradientBoostingClassifier(
            random_state=random_seed,
            learning_rate=0.07,
            max_depth=None,
            max_iter=400,
            l2_regularization=0.05,
            min_samples_leaf=15,
            max_bins=255,
            class_weight="balanced",
        ),
        "logit": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        max_iter=1000,
                        C=2.0,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }
    base_order = list(base_models.keys())
    n_classes = len(RESULT_ORDER)

    def _align_probabilities(model, probabilities: np.ndarray) -> np.ndarray:
        aligned = np.zeros((probabilities.shape[0], n_classes))
        model_classes = le.inverse_transform(model.classes_)
        for idx_src, cls_name in enumerate(model_classes):
            target_idx = RESULT_ORDER.index(cls_name)
            aligned[:, target_idx] = probabilities[:, idx_src]
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return aligned / row_sums

    oof_predictions = {name: np.zeros((len(X_train), n_classes)) for name in base_order}
    test_predictions: Dict[str, np.ndarray] = {}
    base_fitted_models: Dict[str, object] = {}

    n_splits = min(5, max(2, len(X_train) // 120))
    if n_splits < 2:
        n_splits = 2 if len(X_train) >= 120 else 1

    if n_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, val_idx in tscv.split(X_train):
            X_tr_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            for name, estimator in base_models.items():
                model = clone(estimator)
                model.fit(X_tr_fold, y_tr_fold)
                probs = model.predict_proba(X_val_fold)
                aligned = _align_probabilities(model, probs)
                oof_predictions[name][val_idx] = aligned

    for name, estimator in base_models.items():
        model = clone(estimator)
        model.fit(X_train, y_train)
        base_fitted_models[name] = model
        test_predictions[name] = _align_probabilities(model, model.predict_proba(X_test))
        full_train_probs = _align_probabilities(model, model.predict_proba(X_train))
        zero_mask = ~oof_predictions[name].any(axis=1)
        if zero_mask.any():
            oof_predictions[name][zero_mask] = full_train_probs[zero_mask]

    stack_train_components = [oof_predictions[name] for name in base_order]
    stack_test_components = [test_predictions[name] for name in base_order]
    stack_train_components.append(prob_train)
    stack_test_components.append(prob_test)

    if prob_columns:
        book_train = df.iloc[:train_size][prob_columns].fillna(1.0 / len(RESULT_ORDER)).to_numpy()
        book_test = df.iloc[train_size:][prob_columns].fillna(1.0 / len(RESULT_ORDER)).to_numpy()
    else:
        book_train = np.full((train_size, len(RESULT_ORDER)), 1.0 / len(RESULT_ORDER))
        book_test = np.full((len(X_test), len(RESULT_ORDER)), 1.0 / len(RESULT_ORDER))
    stack_train_components.append(book_train)
    stack_test_components.append(book_test)

    edge_subset = [col for col in ["eg_prob_win_edge", "eg_prob_draw_edge", "eg_prob_loss_edge"] if col in df.columns]
    if edge_subset:
        stack_train_components.append(df.iloc[:train_size][edge_subset].fillna(0.0).to_numpy())
        stack_test_components.append(df.iloc[train_size:][edge_subset].fillna(0.0).to_numpy())

    meta_extra_cols = [
        col
        for col in [
            "expected_points_edge",
            "strength_index_diff",
            "points_avg_5_diff",
            "rest_days_diff",
            "team_xg_diff_std_5",
            "opponent_xg_diff_std_5",
            "market_confidence_gap",
        ]
        if col in df.columns
    ]
    if meta_extra_cols:
        stack_train_components.append(df.iloc[:train_size][meta_extra_cols].fillna(0.0).to_numpy())
        stack_test_components.append(df.iloc[train_size:][meta_extra_cols].fillna(0.0).to_numpy())

    stack_train = np.hstack(stack_train_components).astype(float)
    stack_test = np.hstack(stack_test_components).astype(float)

    meta_scaler = StandardScaler()
    stack_train_scaled = meta_scaler.fit_transform(stack_train)
    stack_test_scaled = meta_scaler.transform(stack_test)

    meta_model = LogisticRegression(
        max_iter=1000,
        C=2.0,
        solver="lbfgs",
    )
    meta_model.fit(stack_train_scaled, y_train)
    meta_test_probs = meta_model.predict_proba(stack_test_scaled)
    aligned_proba = _align_probabilities(meta_model, meta_test_probs)

    pred_indices = aligned_proba.argmax(axis=1)
    predictions = [RESULT_ORDER[idx] for idx in pred_indices]
    actual = le.inverse_transform(y_test)

    df_test["predicted"] = predictions
    df_test["actual"] = actual
    df_test["correct"] = df_test["predicted"] == df_test["actual"]

    for name in base_order:
        base_probs = test_predictions[name]
        for idx, cls in enumerate(RESULT_ORDER):
            df_test[f"{name}_prob_{cls}"] = base_probs[:, idx]

    for idx, cls in enumerate(RESULT_ORDER):
        df_test[f"pred_prob_{cls}"] = aligned_proba[:, idx]

    actual_indices = np.array([RESULT_ORDER.index(cls) for cls in actual])
    actual_one_hot = np.zeros_like(aligned_proba)
    actual_one_hot[np.arange(len(actual_indices)), actual_indices] = 1.0

    accuracy = accuracy_score(actual, predictions)
    brier = float(np.mean(np.sum((aligned_proba - actual_one_hot) ** 2, axis=1)))
    logloss = log_loss(actual_indices, aligned_proba, labels=list(range(len(RESULT_ORDER))))

    meta_confidence = aligned_proba.max(axis=1)
    meta_hc_mask = meta_confidence >= 0.6
    if meta_hc_mask.any():
        meta_hc_accuracy = accuracy_score(
            np.array(actual)[meta_hc_mask],
            np.array(predictions)[meta_hc_mask],
        )
        meta_hc_coverage = float(meta_hc_mask.mean())
    else:
        meta_hc_accuracy = float("nan")
        meta_hc_coverage = 0.0

    summary_rows = [
        {"metric": "offline_meta_accuracy", "value": accuracy},
        {"metric": "offline_meta_brier", "value": brier},
        {"metric": "offline_meta_logloss", "value": logloss},
        {"metric": "offline_meta_high_conf_accuracy", "value": meta_hc_accuracy},
        {"metric": "offline_meta_high_conf_coverage", "value": meta_hc_coverage},
    ]

    metrics_records = []
    for season, season_df in df_test.groupby("season"):
        meta_season_accuracy = season_df["correct"].mean()
        metrics_records.append(
            {"season": season, "model": "offline_meta", "accuracy": meta_season_accuracy}
        )

    # Base model diagnostics
    for name in base_order:
        base_probs = test_predictions[name]
        base_pred_indices = base_probs.argmax(axis=1)
        base_predictions = [RESULT_ORDER[idx] for idx in base_pred_indices]
        base_accuracy = accuracy_score(actual, base_predictions)
        base_brier = float(np.mean(np.sum((base_probs - actual_one_hot) ** 2, axis=1)))
        base_logloss = log_loss(actual_indices, base_probs, labels=list(range(len(RESULT_ORDER))))

        if name == "hist":
            summary_rows.extend(
                [
                    {"metric": "offline_gradient_boosting_accuracy", "value": base_accuracy},
                    {"metric": "offline_gradient_boosting_brier", "value": base_brier},
                    {"metric": "offline_gradient_boosting_logloss", "value": base_logloss},
                ]
            )
        else:
            summary_rows.append({"metric": f"offline_{name}_accuracy", "value": base_accuracy})

        hist_cols = [f"{name}_prob_{cls}" for cls in RESULT_ORDER]
        if all(col in df_test.columns for col in hist_cols):
            for season, season_group in df_test.groupby("season"):
                probs_season = season_group[hist_cols].to_numpy()
                preds_season = [RESULT_ORDER[idx] for idx in probs_season.argmax(axis=1)]
                accuracy_season = accuracy_score(season_group["actual"], preds_season)
                metrics_records.append(
                    {
                        "season": season,
                        "model": f"offline_{name}",
                        "accuracy": accuracy_season,
                    }
                )

    summary = pd.DataFrame(summary_rows)

    output_columns = [
        "season",
        "team",
        "opponent",
        "match_id",
        "match_date",
        "venue",
        "predicted",
        "actual",
        "correct",
    ]
    goal_output_cols = [col for col in goal_feature_names if col in df_test.columns]
    predictions_output = df_test[
        output_columns + goal_output_cols + [f"pred_prob_{cls}" for cls in RESULT_ORDER]
    ]

    metrics_df = pd.DataFrame(metrics_records)

    _ensure_directory(paths.outputs_dir)
    perm = permutation_importance(base_fitted_models["hist"], X_test, y_test, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame(
        {
            "feature": all_features,
            "importance": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance", ascending=False)

    # Logistic regression for interpretable win coefficients
    binary_target = (df["result_class"] == "win").astype(int)
    X_log = X.iloc[: len(binary_target)]
    y_log = binary_target
    X_log_train, X_log_test = X_log.iloc[:train_size], X_log.iloc[train_size:]
    y_log_train, y_log_test = y_log.iloc[:train_size], y_log.iloc[train_size:]

    logit_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=1000)),
        ]
    )
    logit_pipeline.fit(X_log_train, y_log_train)
    logit_coef = logit_pipeline.named_steps["logit"].coef_[0]
    logistic_df = pd.DataFrame(
        {
            "feature": all_features,
            "coefficient": logit_coef,
        }
    ).sort_values("coefficient", ascending=False)
    logistic_df["direction"] = logistic_df["coefficient"].apply(
        lambda v: "positive" if v > 0 else "negative"
    )

    _safe_to_csv(predictions_output, paths.outputs_dir / "offline_predictions.csv")
    _safe_to_csv(metrics_df, paths.outputs_dir / "offline_metrics.csv")
    _safe_to_csv(importance_df, paths.outputs_dir / "feature_importance.csv")
    _safe_to_csv(logistic_df, paths.outputs_dir / "feature_effects_win.csv")

    goal_summary = pd.DataFrame(
        [
            {"metric": "goal_model_team_mae", "value": metrics_team["mae"]},
            {"metric": "goal_model_team_rmse", "value": metrics_team["rmse"]},
            {"metric": "goal_model_opponent_mae", "value": metrics_opp["mae"]},
            {"metric": "goal_model_opponent_rmse", "value": metrics_opp["rmse"]},
        ]
    )

    summary = pd.concat([summary, goal_summary], ignore_index=True)

    return {
        "predictions": predictions_output,
        "metrics": metrics_df,
        "summary": summary,
        "feature_importance": importance_df,
        "feature_effects": logistic_df,
        "goal_metrics": {"team": metrics_team, "opponent": metrics_opp},
    }
