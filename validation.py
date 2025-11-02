"""
Walk-forward validation utilities and evaluation metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

RESULT_ORDER = ["win", "draw", "loss"]


@dataclass
class FoldSplit:
    fold_id: int
    train_index: np.ndarray
    valid_index: np.ndarray
    train_seasons: List[str]
    valid_season: str


def walk_forward_splits(fixtures: pd.DataFrame, min_train_seasons: int = 1) -> List[FoldSplit]:
    seasons = sorted(fixtures["season"].unique().tolist())
    splits: List[FoldSplit] = []
    fold_id = 1
    for idx in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:idx]
        valid_season = seasons[idx]
        train_index = fixtures[fixtures["season"].isin(train_seasons)].index.to_numpy()
        valid_index = fixtures[fixtures["season"] == valid_season].index.to_numpy()
        if len(train_index) == 0 or len(valid_index) == 0:
            continue
        splits.append(
            FoldSplit(
                fold_id=fold_id,
                train_index=train_index,
                valid_index=valid_index,
                train_seasons=train_seasons,
                valid_season=valid_season,
            )
        )
        fold_id += 1
    return splits


def _brier_score(y_true: np.ndarray, probas: np.ndarray) -> float:
    n = len(y_true)
    one_hot = np.zeros_like(probas)
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probas - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, probas: np.ndarray, n_bins: int = 10) -> float:
    confidences = probas.max(axis=1)
    predictions = probas.argmax(axis=1)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        acc = np.mean(predictions[mask] == y_true[mask])
        conf = np.mean(confidences[mask])
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def evaluate_metrics(y_true: np.ndarray, probas: np.ndarray) -> dict:
    acc = accuracy_score(y_true, probas.argmax(axis=1))
    brier = _brier_score(y_true, probas)
    ll = log_loss(y_true, probas, labels=list(range(probas.shape[1])))
    ece = expected_calibration_error(y_true, probas)
    draw_rate = float((y_true == 1).mean())
    return {
        "accuracy": float(acc),
        "brier": float(brier),
        "logloss": float(ll),
        "ece": float(ece),
        "draw_rate": draw_rate,
    }
