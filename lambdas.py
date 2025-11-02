from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from workflow import collect_feature_columns


@dataclass
class GoalModelResult:
    regressor: HistGradientBoostingRegressor
    feature_columns: list[str]
    metrics: Dict[str, float]


def _prepare_dataset(dataset_path: Path) -> Tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(dataset_path, parse_dates=["match_date"])
    df = df.sort_values(["match_date", "team"]).reset_index(drop=True)
    df = df[df["team_match_number"] > 5].copy()

    feature_columns = collect_feature_columns(dataset_path)
    X = df[feature_columns].fillna(0.0)
    return df, feature_columns, X


def _train_goal_model(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_depth=6,
        learning_rate=0.05,
        max_iter=600,
        min_samples_leaf=25,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_goal_models(dataset_path: Path) -> Dict[str, GoalModelResult]:
    df, feature_columns, X = _prepare_dataset(dataset_path)

    y_team = df["team_goals"].astype(float)
    y_opp = df["opponent_goals"].astype(float)

    model_team = _train_goal_model(X, y_team)
    model_opp = _train_goal_model(X, y_opp)

    preds_team = np.clip(model_team.predict(X), 1e-6, None)
    preds_opp = np.clip(model_opp.predict(X), 1e-6, None)

    metrics_team = {
        "mae": mean_absolute_error(y_team, preds_team),
        "rmse": mean_squared_error(y_team, preds_team, squared=False),
    }
    metrics_opp = {
        "mae": mean_absolute_error(y_opp, preds_opp),
        "rmse": mean_squared_error(y_opp, preds_opp, squared=False),
    }

    return {
        "team": GoalModelResult(model_team, feature_columns, metrics_team),
        "opponent": GoalModelResult(model_opp, feature_columns, metrics_opp),
    }


if __name__ == "__main__":
    dataset_path = Path("data/processed/epl_unified_team_matches.csv")
    results = train_goal_models(dataset_path)
    for side, result in results.items():
        print(f"Model for {side} goals -> MAE: {result.metrics['mae']:.3f}, RMSE: {result.metrics['rmse']:.3f}")
