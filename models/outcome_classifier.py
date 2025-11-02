"""
Multiclass outcome classifier for fixture-level predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class OutcomeModel:
    classifier: HistGradientBoostingClassifier


DEFAULT_CLF_PARAMS: Dict[str, object] = dict(
    max_depth=6,
    learning_rate=0.05,
    max_iter=600,
    min_samples_leaf=40,
    l2_regularization=0.1,
    class_weight={0: 1.0, 1: 0.7, 2: 1.0},
    random_state=21,
)


def _make_classifier(params: Optional[Dict[str, object]] = None) -> HistGradientBoostingClassifier:
    merged = dict(DEFAULT_CLF_PARAMS)
    if params:
        merged.update(params)
    return HistGradientBoostingClassifier(**merged)


def fit_outcome_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    params: Optional[Dict[str, object]] = None,
) -> OutcomeModel:
    clf = _make_classifier(params)
    clf.fit(X, y)
    return OutcomeModel(classifier=clf)


def predict_proba_outcome(model: OutcomeModel, X: pd.DataFrame) -> np.ndarray:
    return model.classifier.predict_proba(X)


def save_outcome_model(model: OutcomeModel, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.classifier, directory / "outcome_classifier.pkl")


def load_outcome_model(directory: Path) -> OutcomeModel:
    clf = joblib.load(directory / "outcome_classifier.pkl")
    return OutcomeModel(classifier=clf)
