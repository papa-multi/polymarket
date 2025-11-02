"""
Isotonic calibration helpers for multiclass probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

CLASS_NAMES = ["home", "draw", "away"]


@dataclass
class IsotonicCalibrators:
    calibrators: Dict[str, Optional[IsotonicRegression]]


def fit_isotonic_per_class(probas: np.ndarray, y: np.ndarray) -> IsotonicCalibrators:
    """
    Fit isotonic regression per class using one-vs-rest approach.

    Parameters
    ----------
    probas : np.ndarray
        Shape (n_samples, 3) predicted probabilities.
    y : np.ndarray
        Encoded targets (0=home, 1=draw, 2=away).
    """
    calibrators: Dict[str, Optional[IsotonicRegression]] = {}
    for idx, name in enumerate(CLASS_NAMES):
        target = (y == idx).astype(float)
        preds = probas[:, idx]
        unique = np.unique(target)
        if unique.size < 2 or np.allclose(preds, preds[0]):
            calibrators[name] = None
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(preds, target)
        calibrators[name] = ir
    return IsotonicCalibrators(calibrators=calibrators)


def apply_isotonic(probas: np.ndarray, calibrators: IsotonicCalibrators) -> np.ndarray:
    calibrated = probas.copy()
    for idx, name in enumerate(CLASS_NAMES):
        reg = calibrators.calibrators.get(name)
        if reg is None:
            continue
        calibrated[:, idx] = reg.predict(probas[:, idx])
    calibrated = np.clip(calibrated, 1e-6, 1.0)
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    return calibrated
