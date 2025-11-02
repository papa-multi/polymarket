"""
Fixture scoring and forecast utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .calibration import IsotonicCalibrators, apply_isotonic
from .models.goal_poisson import GoalModels, lambdas_to_probas, predict_lambdas
from .models.outcome_classifier import OutcomeModel, predict_proba_outcome


def _stack_features(poisson_probs: np.ndarray, classifier_probs: Optional[np.ndarray]) -> np.ndarray:
    if classifier_probs is None:
        return poisson_probs
    return np.hstack([poisson_probs, classifier_probs])


def score_fixtures(
    fixtures: pd.DataFrame,
    feature_columns: list[str],
    goal_models: GoalModels,
    classifier: Optional[OutcomeModel],
    meta_model,
    calibrators: IsotonicCalibrators,
) -> pd.DataFrame:
    X = fixtures[feature_columns].fillna(0.0)
    lambda_home, lambda_away = predict_lambdas(goal_models, X)
    poisson_probs = lambdas_to_probas(lambda_home, lambda_away)

    classifier_probs = None
    if classifier is not None:
        classifier_probs = predict_proba_outcome(classifier, X)

    if meta_model is not None and classifier_probs is not None:
        meta_features = np.hstack([poisson_probs, classifier_probs])
        final_probs = meta_model.predict_proba(meta_features)
    elif meta_model is not None:
        final_probs = meta_model.predict_proba(poisson_probs)
    else:
        final_probs = poisson_probs

    calibrated = apply_isotonic(final_probs, calibrators)

    entropy = -(calibrated * np.log(calibrated + 1e-12)).sum(axis=1)
    max_entropy = np.log(calibrated.shape[1])
    confidence = 1.0 - entropy / max_entropy

    result = fixtures[
        ["season", "match_date", "home_team", "away_team", "match_id", "home_odds_win", "home_odds_draw", "home_odds_loss"]
    ].copy()
    result["p_home_win"] = calibrated[:, 0]
    result["p_draw"] = calibrated[:, 1]
    result["p_away_win"] = calibrated[:, 2]
    result["model_confidence"] = confidence
    return result
