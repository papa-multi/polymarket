"""
Goal expectation models (per-team Poisson) for fixture forecasting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

EPS = 1e-6


@dataclass
class GoalModels:
    home: HistGradientBoostingRegressor
    away: HistGradientBoostingRegressor


DEFAULT_PARAMS: Dict[str, object] = dict(
    loss="poisson",
    max_depth=6,
    learning_rate=0.05,
    max_iter=800,
    min_samples_leaf=20,
    l2_regularization=0.05,
    random_state=42,
)


def _make_regressor(params: Optional[Dict[str, object]] = None) -> HistGradientBoostingRegressor:
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    return HistGradientBoostingRegressor(**merged)


def fit_goal_models(
    X: pd.DataFrame,
    y_home: pd.Series,
    y_away: pd.Series,
    params: Optional[Dict[str, object]] = None,
) -> GoalModels:
    model_home = _make_regressor(params)
    model_away = _make_regressor(params)
    model_home.fit(X, y_home)
    model_away.fit(X, y_away)
    return GoalModels(home=model_home, away=model_away)


def predict_lambdas(models: GoalModels, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    lambda_home = np.clip(models.home.predict(X), EPS, None)
    lambda_away = np.clip(models.away.predict(X), EPS, None)
    return lambda_home, lambda_away


def lambdas_to_probas(
    lambda_home: np.ndarray, lambda_away: np.ndarray, max_goals: int = 8
) -> np.ndarray:
    """Convert goal lambdas to match outcome probabilities via independent Poisson."""
    lambda_home = np.clip(lambda_home, EPS, None)
    lambda_away = np.clip(lambda_away, EPS, None)
    probs = np.zeros((len(lambda_home), 3), dtype=float)  # [home, draw, away]

    goal_range = np.arange(0, max_goals + 1)
    from math import factorial as fact

    factorial = np.array([fact(k) for k in goal_range], dtype=float)

    for idx, (lam_h, lam_a) in enumerate(zip(lambda_home, lambda_away)):
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        for gh in goal_range:
            prob_h = np.exp(-lam_h) * (lam_h**gh) / factorial[gh]
            for ga in goal_range:
                prob_a = np.exp(-lam_a) * (lam_a**ga) / factorial[ga]
                joint = prob_h * prob_a
                if gh > ga:
                    p_home += joint
                elif gh < ga:
                    p_away += joint
                else:
                    p_draw += joint
        remainder = max(0.0, 1.0 - (p_home + p_draw + p_away))
        p_draw += remainder  # assign tail mass to draw
        probs[idx, 0] = p_home
        probs[idx, 1] = p_draw
        probs[idx, 2] = p_away

    probs = np.clip(probs, EPS, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def save_goal_models(models: GoalModels, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(models.home, directory / "goals_poisson_home.pkl")
    joblib.dump(models.away, directory / "goals_poisson_away.pkl")


def load_goal_models(directory: Path) -> GoalModels:
    model_home = joblib.load(directory / "goals_poisson_home.pkl")
    model_away = joblib.load(directory / "goals_poisson_away.pkl")
    return GoalModels(home=model_home, away=model_away)
