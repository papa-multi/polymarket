"""
Rebalanced modelling pipeline built on fixture-level features, calibrated probabilities,
and bookmaker-aware blending. Produces historical evaluation and upcoming fixture forecasts.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from .config_loader import FootballConfig
from .fixtures import FixturePaths, build_fixture_table
from .goal_model import poisson_outcome_probabilities, train_goal_expectations


TARGET_CLASSES: Tuple[str, ...] = ("win", "draw", "loss")
EPS = 1e-6


# ---------------------------------------------------------------------------
# Team history helpers
# ---------------------------------------------------------------------------


@dataclass
class TeamMatchRecord:
    goals_for: float
    goals_against: float
    xg_for: float
    xg_against: float
    points: float
    win: float
    date: pd.Timestamp


@dataclass
class TeamState:
    overall: Deque[TeamMatchRecord]
    home: Deque[TeamMatchRecord]
    away: Deque[TeamMatchRecord]
    elo: float = 1500.0
    last_match_date: Optional[pd.Timestamp] = None

    def append(self, record: TeamMatchRecord, is_home: bool) -> None:
        self.overall.append(record)
        if is_home:
            self.home.append(record)
        else:
            self.away.append(record)
        self.last_match_date = record.date


def _default_state() -> TeamState:
    return TeamState(overall=deque(maxlen=200), home=deque(maxlen=200), away=deque(maxlen=200))


def _aggregate(records: Iterable[TeamMatchRecord], window: int) -> Dict[str, float]:
    subset = list(records)[-window:] if window else list(records)
    if not subset:
        return {
            "goals_avg": 0.0,
            "goals_against_avg": 0.0,
            "goal_diff_avg": 0.0,
            "xg_avg": 0.0,
            "xg_against_avg": 0.0,
            "finishing": 1.0,
            "defending": 1.0,
            "points_avg": 0.0,
            "win_rate": 0.0,
        }
    n = len(subset)
    gf = sum(r.goals_for for r in subset)
    ga = sum(r.goals_against for r in subset)
    xgf = sum(r.xg_for for r in subset)
    xga = sum(r.xg_against for r in subset)
    pts = sum(r.points for r in subset)
    wins = sum(r.win for r in subset)
    return {
        "goals_avg": gf / n,
        "goals_against_avg": ga / n,
        "goal_diff_avg": (gf - ga) / n,
        "xg_avg": xgf / n,
        "xg_against_avg": xga / n,
        "finishing": gf / max(xgf, EPS),
        "defending": ga / max(xga, EPS),
        "points_avg": pts / n,
        "win_rate": wins / n,
    }


def _result_to_points_outcome(label: str) -> Tuple[float, float]:
    if label == "win":
        return 3.0, 1.0
    if label == "draw":
        return 1.0, 0.5
    if label == "loss":
        return 0.0, 0.0
    raise ValueError(f"Unknown label {label}")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _compute_team_features(state: TeamState, match_date: pd.Timestamp, is_home: bool) -> Dict[str, float]:
    overall = state.overall
    home_or_away = state.home if is_home else state.away

    agg_overall_5 = _aggregate(overall, 5)
    agg_overall_10 = _aggregate(overall, 10)
    agg_side_5 = _aggregate(home_or_away, 5)

    rest_days = (
        (match_date - state.last_match_date).days if state.last_match_date is not None else 7.0
    )

    prefix = "home" if is_home else "away"
    features = {
        f"{prefix}_matches_played": float(len(overall)),
        f"{prefix}_side_matches_played": float(len(home_or_away)),
        f"{prefix}_goals_avg_last5": agg_overall_5["goals_avg"],
        f"{prefix}_goals_against_avg_last5": agg_overall_5["goals_against_avg"],
        f"{prefix}_goal_diff_avg_last5": agg_overall_5["goal_diff_avg"],
        f"{prefix}_xg_avg_last5": agg_overall_5["xg_avg"],
        f"{prefix}_xg_against_avg_last5": agg_overall_5["xg_against_avg"],
        f"{prefix}_finishing_last5": agg_overall_5["finishing"],
        f"{prefix}_defending_last5": agg_overall_5["defending"],
        f"{prefix}_points_avg_last5": agg_overall_5["points_avg"],
        f"{prefix}_points_avg_last10": agg_overall_10["points_avg"],
        f"{prefix}_win_rate_last5": agg_overall_5["win_rate"],
        f"{prefix}_side_goals_avg_last5": agg_side_5["goals_avg"],
        f"{prefix}_side_win_rate_last5": agg_side_5["win_rate"],
        f"{prefix}_elo": state.elo,
        f"{prefix}_rest_days": float(rest_days),
    }
    return features


def _update_elo(home_state: TeamState, away_state: TeamState, label: str, k: float = 24.0) -> None:
    result_map = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    home_result = result_map[label]
    expected_home = 1.0 / (1.0 + 10.0 ** ((away_state.elo - home_state.elo) / 400.0))
    home_state.elo += k * (home_result - expected_home)
    away_state.elo += k * ((1.0 - home_result) - (1.0 - expected_home))


def build_fixture_features(fixtures: pd.DataFrame) -> pd.DataFrame:
    fixtures = fixtures.sort_values("match_date").reset_index(drop=True)
    team_states: Dict[str, TeamState] = defaultdict(_default_state)
    season_matchday: Dict[str, int] = defaultdict(int)

    records: List[Dict[str, float]] = []

    for _, row in fixtures.iterrows():
        season = row["season"]
        match_date = pd.to_datetime(row["match_date"])
        season_matchday[season] += 1
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_state = team_states[home_team]
        away_state = team_states[away_team]

        home_feats = _compute_team_features(home_state, match_date, is_home=True)
        away_feats = _compute_team_features(away_state, match_date, is_home=False)

        market_probs = np.array(
            [
                row.get("prob_home_win_market", np.nan),
                row.get("prob_draw_market", np.nan),
                row.get("prob_away_win_market", np.nan),
            ],
            dtype=float,
        )
        if np.any(np.isnan(market_probs)) or market_probs.sum() <= 0:
            market_probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        else:
            market_probs = market_probs / market_probs.sum()

        feature_row: Dict[str, float] = {
            "match_id": row["match_id"],
            "season": season,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "matchday": season_matchday[season],
            "season_int": float(season.split("-")[0]),
            "label": row.get("result_class"),
            "home_goals": row.get("home_goals", np.nan),
            "away_goals": row.get("away_goals", np.nan),
            "home_xg": row.get("home_xg", np.nan),
            "away_xg": row.get("away_xg", np.nan),
            "home_rest_days_fixture": row.get("home_rest_days", np.nan),
            "away_rest_days_fixture": row.get("away_rest_days", np.nan),
            "prob_home_win_market": market_probs[0],
            "prob_draw_market": market_probs[1],
            "prob_away_win_market": market_probs[2],
        }
        feature_row.update(home_feats)
        feature_row.update(away_feats)

        # Differences
        feature_row["elo_diff"] = home_feats["home_elo"] - away_feats["away_elo"]
        feature_row["goals_avg_diff_last5"] = (
            home_feats["home_goals_avg_last5"] - away_feats["away_goals_avg_last5"]
        )
        feature_row["xg_avg_diff_last5"] = (
            home_feats["home_xg_avg_last5"] - away_feats["away_xg_avg_last5"]
        )
        feature_row["finishing_diff_last5"] = (
            home_feats["home_finishing_last5"] - away_feats["away_finishing_last5"]
        )
        feature_row["points_avg_diff_last5"] = (
            home_feats["home_points_avg_last5"] - away_feats["away_points_avg_last5"]
        )

        records.append(feature_row)

        # Update states only if result known
        label = row.get("result_class")
        if isinstance(label, str) and label in {"win", "draw", "loss"}:
            home_goals = float(row.get("home_goals", np.nan))
            away_goals = float(row.get("away_goals", np.nan))
            home_xg = float(row.get("home_xg", np.nan))
            away_xg = float(row.get("away_xg", np.nan))

            home_points, home_win = _result_to_points_outcome(label)
            away_label = "win" if label == "loss" else "loss" if label == "win" else "draw"
            away_points, away_win = _result_to_points_outcome(away_label)

            home_record = TeamMatchRecord(
                goals_for=home_goals,
                goals_against=away_goals,
                xg_for=home_xg if np.isfinite(home_xg) else home_goals,
                xg_against=away_xg if np.isfinite(away_xg) else away_goals,
                points=home_points,
                win=home_win,
                date=match_date,
            )
            away_record = TeamMatchRecord(
                goals_for=away_goals,
                goals_against=home_goals,
                xg_for=away_xg if np.isfinite(away_xg) else away_goals,
                xg_against=home_xg if np.isfinite(home_xg) else home_goals,
                points=away_points,
                win=away_win,
                date=match_date,
            )
            home_state.append(home_record, is_home=True)
            away_state.append(away_record, is_home=False)
            _update_elo(home_state, away_state, label)

    feature_df = pd.DataFrame(records)

    # Replace NaNs for rest days using historical estimates
    for side in ("home", "away"):
        est_col = f"{side}_rest_days"
        fixture_col = f"{side}_rest_days_fixture"
        feature_df[est_col] = feature_df[fixture_col].fillna(feature_df[est_col])
        feature_df[est_col] = feature_df[est_col].fillna(7.0)
    feature_df.drop(columns=["home_rest_days_fixture", "away_rest_days_fixture"], inplace=True)

    return feature_df


# ---------------------------------------------------------------------------
# Modelling utilities
# ---------------------------------------------------------------------------


def _encode_labels(series: pd.Series) -> np.ndarray:
    mapping = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
    return series.map(mapping).to_numpy(dtype=int)


def _compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _blend_probabilities(model_probs: np.ndarray, market_probs: np.ndarray, weight: float) -> np.ndarray:
    weight = np.clip(weight, 0.0, 1.0)
    eps = 1e-6
    base = np.clip(model_probs, eps, 1.0)
    market = np.clip(market_probs, eps, 1.0)
    blended = (base ** weight) * (market ** (1.0 - weight))
    blended /= blended.sum(axis=1, keepdims=True)
    return blended


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def _walk_forward_splits(
    ordered_seasons: Sequence[str],
    *,
    minimum_training_seasons: int = 1,
) -> List[Tuple[List[str], List[str]]]:
    """Return walk-forward splits using chronological season order."""
    splits: List[Tuple[List[str], List[str]]] = []
    for idx in range(minimum_training_seasons, len(ordered_seasons)):
        train = list(ordered_seasons[:idx])
        valid = [ordered_seasons[idx]]
        if train and valid:
            splits.append((train, valid))
    return splits


def _meta_feature_frame(
    model_probs: np.ndarray,
    goal_probs: np.ndarray,
    index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Construct a DataFrame of stacking features without bookmaker inputs."""
    data = {
        "model_prob_win": model_probs[:, 0],
        "model_prob_draw": model_probs[:, 1],
        "model_prob_loss": model_probs[:, 2],
        "goal_prob_win": goal_probs[:, 0],
        "goal_prob_draw": goal_probs[:, 1],
        "goal_prob_loss": goal_probs[:, 2],
        "model_minus_goal_win": model_probs[:, 0] - goal_probs[:, 0],
        "model_minus_goal_draw": model_probs[:, 1] - goal_probs[:, 1],
        "model_minus_goal_loss": model_probs[:, 2] - goal_probs[:, 2],
    }
    return pd.DataFrame(data, index=index)


def _validation_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    """Compute accuracy/Brier/logloss over validation predictions."""
    if predictions.empty:
        return {"validation_accuracy": float("nan"), "validation_brier": float("nan"), "validation_logloss": float("nan")}

    y_true = predictions["label"].map({cls: idx for idx, cls in enumerate(TARGET_CLASSES)}).to_numpy()
    prob_cols = ["prob_home_win", "prob_draw", "prob_away_win"]
    proba = predictions[prob_cols].to_numpy(dtype=float)
    accuracy = accuracy_score(y_true, np.argmax(proba, axis=1))
    brier = _compute_brier(proba, y_true)
    ll = log_loss(y_true, proba, labels=list(range(len(TARGET_CLASSES))))
    return {
        "validation_accuracy": float(accuracy),
        "validation_brier": float(brier),
        "validation_logloss": float(ll),
    }


def train_rebalanced_model(fixtures_path: Path, config: FootballConfig, outputs_dir: Path) -> Dict[str, pd.DataFrame]:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    fixtures = pd.read_csv(fixtures_path, parse_dates=["match_date"])
    feature_df = build_fixture_features(fixtures)
    feature_df.to_csv(outputs_dir / "fixture_features.csv", index=False)

    played_mask = (
        feature_df["home_goals"].notna()
        & feature_df["away_goals"].notna()
        & feature_df["home_xg"].notna()
        & feature_df["away_xg"].notna()
    )
    labeled_df = feature_df[feature_df["label"].isin(TARGET_CLASSES) & played_mask].copy()
    future_df = feature_df[~played_mask].copy()

    exclude_cols = {
        "home_team",
        "away_team",
        "home_team_encoded",
        "away_team_encoded",
        "home_rest_days_fixture",
        "away_rest_days_fixture",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
    }
    feature_columns = [
        col
        for col in feature_df.columns
        if col.startswith(
            (
                "home_",
                "away_",
                "elo_",
                "goals_",
                "xg_",
                "points_",
                "finishing_",
                "diff_",
                "matchday",
                "season_int",
            )
        )
        and col not in exclude_cols
    ]
    feature_columns = sorted(set(feature_columns))

    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    available_seasons = list(dict.fromkeys(s["label"] for s in config.seasons))
    historical_seasons = [season for season in available_seasons if season in labeled_df["season"].unique()]
    if not historical_seasons:
        historical_seasons = sorted(labeled_df["season"].unique())

    splits = _walk_forward_splits(historical_seasons, minimum_training_seasons=1)
    if not splits and len(historical_seasons) >= 2:
        splits = [(historical_seasons[:-1], [historical_seasons[-1]])]

    base_template = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=600,
        l2_regularization=0.05,
        min_samples_leaf=12,
        random_state=int(config.raw.get("project", {}).get("random_seed", 21)),
        max_bins=255,
    )
    fold_predictions: List[pd.DataFrame] = []
    stacking_frames: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, float]] = []
    goal_metrics_home: List[Dict[str, float]] = []
    goal_metrics_away: List[Dict[str, float]] = []

    for fold_idx, (train_seasons, valid_seasons) in enumerate(splits, start=1):
        train_df = labeled_df[labeled_df["season"].isin(train_seasons)].copy()
        valid_df = labeled_df[labeled_df["season"].isin(valid_seasons)].copy()
        if train_df.empty or valid_df.empty:
            continue

        train_X = train_df[feature_columns].fillna(0.0).to_numpy(dtype=float)
        valid_X = valid_df[feature_columns].fillna(0.0).to_numpy(dtype=float)
        y_train = _encode_labels(train_df["label"])
        y_valid = _encode_labels(valid_df["label"])

        split_count = min(5, max(2, len(train_df) // 160))
        tscv = TimeSeriesSplit(n_splits=split_count)
        calibrated_clf = CalibratedClassifierCV(
            estimator=HistGradientBoostingClassifier(**base_template.get_params()),
            method="sigmoid",
            cv=tscv,
        )
        calibrated_clf.fit(train_X, y_train)
        calibrated_train = calibrated_clf.predict_proba(train_X)
        calibrated_valid = calibrated_clf.predict_proba(valid_X)
        calibrated_train = np.clip(calibrated_train, 0.02, 0.98)
        calibrated_valid = np.clip(calibrated_valid, 0.02, 0.98)
        calibrated_train = calibrated_train / calibrated_train.sum(axis=1, keepdims=True)
        calibrated_valid = calibrated_valid / calibrated_valid.sum(axis=1, keepdims=True)

        goal_features_train = train_df[feature_columns].fillna(0.0)
        goal_features_valid = valid_df[feature_columns].fillna(0.0)

        lambda_home_train, lambda_home_valid, metrics_home = train_goal_expectations(
            goal_features_train,
            train_df["home_goals"].astype(float),
            goal_features_valid,
        )
        lambda_away_train, lambda_away_valid, metrics_away = train_goal_expectations(
            goal_features_train,
            train_df["away_goals"].astype(float),
            goal_features_valid,
        )
        goal_metrics_home.append(metrics_home)
        goal_metrics_away.append(metrics_away)

        goal_probs_train = poisson_outcome_probabilities(lambda_home_train, lambda_away_train)
        goal_probs_valid = poisson_outcome_probabilities(lambda_home_valid, lambda_away_valid)
        goal_probs_train = np.clip(goal_probs_train, 0.02, 0.98)
        goal_probs_valid = np.clip(goal_probs_valid, 0.02, 0.98)
        goal_probs_train = goal_probs_train / goal_probs_train.sum(axis=1, keepdims=True)
        goal_probs_valid = goal_probs_valid / goal_probs_valid.sum(axis=1, keepdims=True)

        market_valid = valid_df[
            ["prob_home_win_market", "prob_draw_market", "prob_away_win_market"]
        ].to_numpy(dtype=float)

        train_meta = _meta_feature_frame(calibrated_train, goal_probs_train)
        valid_meta = _meta_feature_frame(calibrated_valid, goal_probs_valid)

        meta_model = LogisticRegression(max_iter=1000, solver="lbfgs")
        meta_model.fit(train_meta, y_train)
        final_valid_probs = meta_model.predict_proba(valid_meta)
        final_valid_probs = np.clip(final_valid_probs, 0.02, 0.98)
        final_valid_probs = final_valid_probs / final_valid_probs.sum(axis=1, keepdims=True)

        predictions = pd.DataFrame(
            {
                "season": valid_df["season"].to_numpy(),
                "match_date": valid_df["match_date"].to_numpy(),
                "home_team": valid_df["home_team"].to_numpy(),
                "away_team": valid_df["away_team"].to_numpy(),
                "label": valid_df["label"].to_numpy(),
                "prob_home_win": final_valid_probs[:, 0],
                "prob_draw": final_valid_probs[:, 1],
                "prob_away_win": final_valid_probs[:, 2],
                "model_prob_home_win": calibrated_valid[:, 0],
                "model_prob_draw": calibrated_valid[:, 1],
                "model_prob_away_win": calibrated_valid[:, 2],
                "goal_prob_home_win": goal_probs_valid[:, 0],
                "goal_prob_draw": goal_probs_valid[:, 1],
                "goal_prob_away_win": goal_probs_valid[:, 2],
                "market_prob_home_win": market_valid[:, 0],
                "market_prob_draw": market_valid[:, 1],
                "market_prob_away_win": market_valid[:, 2],
            }
        )
        predictions["predicted"] = [
            TARGET_CLASSES[idx] for idx in np.argmax(final_valid_probs, axis=1)
        ]
        predictions["correct"] = predictions["predicted"] == predictions["label"]
        predictions["fold"] = fold_idx
        fold_predictions.append(predictions)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "train_seasons": ", ".join(train_seasons),
                "valid_season": ", ".join(valid_seasons),
                "accuracy": accuracy_score(y_valid, np.argmax(final_valid_probs, axis=1)),
                "brier": _compute_brier(final_valid_probs, y_valid),
                "logloss": log_loss(y_valid, final_valid_probs, labels=list(range(len(TARGET_CLASSES)))),
            }
        )

        valid_meta["label"] = predictions["label"].to_numpy()
        stacking_frames.append(valid_meta)

    validation_predictions = (
        pd.concat(fold_predictions, ignore_index=True) if fold_predictions else pd.DataFrame()
    )

    stacking_df = pd.concat(stacking_frames, ignore_index=True) if stacking_frames else pd.DataFrame()
    meta_feature_columns = [col for col in stacking_df.columns if col != "label"]

    # Train final stacking model on out-of-fold validation features
    if not stacking_df.empty:
        meta_model_final = LogisticRegression(max_iter=2000, solver="lbfgs")
        meta_model_final.fit(stacking_df[meta_feature_columns], _encode_labels(stacking_df["label"]))
    else:
        meta_model_final = LogisticRegression(max_iter=2000, solver="lbfgs")

    # Refit base model on all labelled data
    all_train_X = labeled_df[feature_columns].fillna(0.0).to_numpy(dtype=float)
    all_y = _encode_labels(labeled_df["label"])

    full_split_count = min(5, max(2, len(labeled_df) // 160))
    tscv_full = TimeSeriesSplit(n_splits=full_split_count)
    calibrated_full = CalibratedClassifierCV(
        estimator=HistGradientBoostingClassifier(**base_template.get_params()),
        method="sigmoid",
        cv=tscv_full,
    )
    calibrated_full.fit(all_train_X, all_y)
    calibrated_all = calibrated_full.predict_proba(all_train_X)
    calibrated_all = np.clip(calibrated_all, 0.02, 0.98)
    calibrated_all = calibrated_all / calibrated_all.sum(axis=1, keepdims=True)

    goal_lambda_all_home, goal_lambda_future_home, metrics_home_all = train_goal_expectations(
        labeled_df[feature_columns].fillna(0.0),
        labeled_df["home_goals"].astype(float),
        future_df[feature_columns].fillna(0.0) if not future_df.empty else pd.DataFrame(columns=feature_columns),
    )
    goal_lambda_all_away, goal_lambda_future_away, metrics_away_all = train_goal_expectations(
        labeled_df[feature_columns].fillna(0.0),
        labeled_df["away_goals"].astype(float),
        future_df[feature_columns].fillna(0.0) if not future_df.empty else pd.DataFrame(columns=feature_columns),
    )

    goal_metrics_home.append(metrics_home_all)
    goal_metrics_away.append(metrics_away_all)

    goal_probs_all = poisson_outcome_probabilities(goal_lambda_all_home, goal_lambda_all_away)
    goal_probs_all = np.clip(goal_probs_all, 0.02, 0.98)
    goal_probs_all = goal_probs_all / goal_probs_all.sum(axis=1, keepdims=True)
    market_all = labeled_df[
        ["prob_home_win_market", "prob_draw_market", "prob_away_win_market"]
    ].to_numpy(dtype=float)

    stacking_all = _meta_feature_frame(calibrated_all, goal_probs_all, index=labeled_df.index)

    if meta_feature_columns and not stacking_df.empty:
        historical_probs = meta_model_final.predict_proba(stacking_all[meta_feature_columns])
    else:
        historical_probs = calibrated_all
    historical_probs = np.clip(historical_probs, 0.02, 0.98)
    historical_probs = historical_probs / historical_probs.sum(axis=1, keepdims=True)

    historical_output = labeled_df[
        ["season", "match_date", "home_team", "away_team", "label"]
    ].copy()
    historical_output["prob_home_win"] = historical_probs[:, 0]
    historical_output["prob_draw"] = historical_probs[:, 1]
    historical_output["prob_away_win"] = historical_probs[:, 2]
    historical_output["market_prob_home_win"] = market_all[:, 0]
    historical_output["market_prob_draw"] = market_all[:, 1]
    historical_output["market_prob_away_win"] = market_all[:, 2]
    historical_output["model_vs_market_home"] = historical_output["prob_home_win"] - historical_output["market_prob_home_win"]
    historical_output["model_vs_market_draw"] = historical_output["prob_draw"] - historical_output["market_prob_draw"]
    historical_output["model_vs_market_away"] = historical_output["prob_away_win"] - historical_output["market_prob_away_win"]
    historical_output["predicted"] = [
        TARGET_CLASSES[idx] for idx in np.argmax(historical_probs, axis=1)
    ]
    historical_output["correct"] = historical_output["predicted"] == historical_output["label"]
    historical_output.to_csv(outputs_dir / "final_predictions.csv", index=False)

    # Future forecasts
    if future_df.empty:
        future_output = pd.DataFrame(
            columns=[
                "match_date",
                "match_id",
                "season",
                "home_team",
                "away_team",
                "prob_home_win",
                "prob_draw",
                "prob_away_win",
            ]
        )
    else:
        future_X = future_df[feature_columns].fillna(0.0).to_numpy(dtype=float)
        calibrated_future = calibrated_full.predict_proba(future_X)
        calibrated_future = np.clip(calibrated_future, 0.02, 0.98)
        calibrated_future = calibrated_future / calibrated_future.sum(axis=1, keepdims=True)
        goal_probs_future = poisson_outcome_probabilities(goal_lambda_future_home, goal_lambda_future_away)
        goal_probs_future = np.clip(goal_probs_future, 0.02, 0.98)
        goal_probs_future = goal_probs_future / goal_probs_future.sum(axis=1, keepdims=True)
        market_future = future_df[
            ["prob_home_win_market", "prob_draw_market", "prob_away_win_market"]
        ].to_numpy(dtype=float)

        if meta_feature_columns and not stacking_df.empty:
            future_meta = _meta_feature_frame(calibrated_future, goal_probs_future)
            final_future_probs = meta_model_final.predict_proba(future_meta[meta_feature_columns])
        else:
            future_meta = _meta_feature_frame(calibrated_future, goal_probs_future)
            final_future_probs = calibrated_future

        final_future_probs = np.clip(final_future_probs, 0.02, 0.98)
        final_future_probs = final_future_probs / final_future_probs.sum(axis=1, keepdims=True)

        confidence = final_future_probs.max(axis=1)
        future_output = future_df[
            ["match_date", "match_id", "season", "home_team", "away_team"]
        ].copy()
        future_output["prob_home_win"] = final_future_probs[:, 0]
        future_output["prob_draw"] = final_future_probs[:, 1]
        future_output["prob_away_win"] = final_future_probs[:, 2]
        future_output["model_prob_home_win"] = calibrated_future[:, 0]
        future_output["model_prob_draw"] = calibrated_future[:, 1]
        future_output["model_prob_away_win"] = calibrated_future[:, 2]
        future_output["goal_lambda_home"] = goal_lambda_future_home
        future_output["goal_lambda_away"] = goal_lambda_future_away
        future_output["goal_prob_home_win"] = goal_probs_future[:, 0]
        future_output["goal_prob_draw"] = goal_probs_future[:, 1]
        future_output["goal_prob_away_win"] = goal_probs_future[:, 2]
        future_output["prob_home_win_market"] = market_future[:, 0]
        future_output["prob_draw_market"] = market_future[:, 1]
        future_output["prob_away_win_market"] = market_future[:, 2]
        future_output["model_vs_market_home"] = future_output["prob_home_win"] - future_output["prob_home_win_market"]
        future_output["model_vs_market_draw"] = future_output["prob_draw"] - future_output["prob_draw_market"]
        future_output["model_vs_market_away"] = future_output["prob_away_win"] - future_output["prob_away_win_market"]
        future_output["confidence"] = confidence
        future_output["elo_diff"] = future_df["elo_diff"].to_numpy()
        future_output["goals_avg_diff_last5"] = future_df["goals_avg_diff_last5"].to_numpy()
        future_output["finishing_diff_last5"] = future_df["finishing_diff_last5"].to_numpy()
        future_output["home_points_avg_last5"] = future_df.get("home_points_avg_last5", pd.Series(np.nan, index=future_df.index)).to_numpy()
        future_output["away_points_avg_last5"] = future_df.get("away_points_avg_last5", pd.Series(np.nan, index=future_df.index)).to_numpy()
        future_output["home_finishing_last5"] = future_df.get("home_finishing_last5", pd.Series(np.nan, index=future_df.index)).to_numpy()
        future_output["away_finishing_last5"] = future_df.get("away_finishing_last5", pd.Series(np.nan, index=future_df.index)).to_numpy()
        future_output.sort_values("match_date", inplace=True)

        team_dates = pd.concat(
            [
                future_output[["match_date", "home_team"]].rename(columns={"home_team": "team"}),
                future_output[["match_date", "away_team"]].rename(columns={"away_team": "team"}),
            ],
            ignore_index=True,
        )
        future_output.sort_values("match_date", inplace=True)
        selected_indices: List[int] = []
        teams_seen: set[str] = set()
        for idx, row in future_output.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            if home in teams_seen or away in teams_seen:
                continue
            selected_indices.append(idx)
            teams_seen.add(home)
            teams_seen.add(away)
        future_output = future_output.loc[selected_indices].sort_values("match_date").reset_index(drop=True)

    future_output.to_csv(outputs_dir / "forecast_clean.csv", index=False)
    high_conf = future_output[future_output.get("confidence", 0.0) >= 0.6].copy()
    high_conf.to_csv(outputs_dir / "forecast_high_conf.csv", index=False)

    diagnostics = future_output.copy()
    diagnostics["draw_heavy_flag"] = diagnostics["prob_draw"] >= 0.4
    diagnostics["home_trending_up"] = diagnostics["home_points_avg_last5"]
    diagnostics["away_trending_up"] = diagnostics["away_points_avg_last5"]
    diagnostics["finishing_gap"] = diagnostics["home_finishing_last5"] - diagnostics["away_finishing_last5"]
    diagnostics["market_edge_home"] = diagnostics["model_vs_market_home"]
    diagnostics["market_edge_draw"] = diagnostics["model_vs_market_draw"]
    diagnostics["market_edge_away"] = diagnostics["model_vs_market_away"]
    diagnostics.to_csv(outputs_dir / "forecast_diagnostics.csv", index=False)

    metrics = _validation_metrics(validation_predictions)
    summary_rows = [
        {"metric": "validation_accuracy", "value": metrics["validation_accuracy"]},
        {"metric": "validation_brier", "value": metrics["validation_brier"]},
        {"metric": "validation_logloss", "value": metrics["validation_logloss"]},
    ]

    if goal_metrics_home:
        avg_home_mae = float(np.mean([m["mae"] for m in goal_metrics_home if m]))
        avg_home_rmse = float(np.mean([m["rmse"] for m in goal_metrics_home if m]))
        avg_away_mae = float(np.mean([m["mae"] for m in goal_metrics_away if m]))
        avg_away_rmse = float(np.mean([m["rmse"] for m in goal_metrics_away if m]))
        summary_rows.extend(
            [
                {"metric": "goal_home_mae", "value": avg_home_mae},
                {"metric": "goal_home_rmse", "value": avg_home_rmse},
                {"metric": "goal_away_mae", "value": avg_away_mae},
                {"metric": "goal_away_rmse", "value": avg_away_rmse},
            ]
        )

    if fold_metrics:
        mean_accuracy = float(np.mean([m["accuracy"] for m in fold_metrics]))
        mean_brier = float(np.mean([m["brier"] for m in fold_metrics]))
        mean_logloss = float(np.mean([m["logloss"] for m in fold_metrics]))
        summary_rows.extend(
            [
                {"metric": "fold_mean_accuracy", "value": mean_accuracy},
                {"metric": "fold_mean_brier", "value": mean_brier},
                {"metric": "fold_mean_logloss", "value": mean_logloss},
            ]
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outputs_dir / "summary.csv", index=False)

    return {
        "summary": summary,
        "validation_predictions": validation_predictions,
        "future_predictions": future_output,
    }


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def run_rebalanced_pipeline(config: FootballConfig, outputs_dir: Path) -> Dict[str, pd.DataFrame]:
    fixture_paths = FixturePaths.default(Path(__file__).resolve().parent)
    fixtures_path = build_fixture_table(fixture_paths)
    return train_rebalanced_model(fixtures_path, config, outputs_dir)
