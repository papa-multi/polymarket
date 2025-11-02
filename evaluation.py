"""
Helper utilities to convert predictions into DataFrames and compute metrics.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .model import MatchPrediction, ResultClass


RESULT_ORDER: List[ResultClass] = ["win", "draw", "loss"]


def predictions_to_dataframe(predictions: Iterable[MatchPrediction]) -> pd.DataFrame:
    flat_records: List[Dict[str, object]] = []
    for p in predictions:
        record = {
            "season": p.season,
            "team": p.team,
            "opponent": p.opponent,
            "team_match_number": p.match_index,
            "match_id": p.match_id,
            "match_date": p.match_date,
            "predicted": p.predicted,
            "actual": p.actual,
            "correct": p.correct,
        }
        for key, value in p.model_scores.items():
            record[f"model_score_{key}"] = value
        for key, value in p.model_probabilities.items():
            record[f"model_prob_{key}"] = value
        if p.expected_goal_probabilities:
            for key, value in p.expected_goal_probabilities.items():
                record[f"eg_prob_{key}"] = value
        if p.bookmaker_probabilities:
            valid_probs = {k: v for k, v in p.bookmaker_probabilities.items() if pd.notna(v)}
            for key, value in p.bookmaker_probabilities.items():
                record[f"book_prob_{key}"] = value
            record["bookmaker_pick"] = (
                max(valid_probs, key=valid_probs.get) if valid_probs else None
            )
        else:
            record["bookmaker_pick"] = None
        flat_records.append(record)
    if not flat_records:
        base_cols = [
            "season",
            "team",
            "opponent",
            "team_match_number",
            "match_id",
            "match_date",
            "predicted",
            "actual",
            "correct",
        ]
        prob_cols = [f"model_prob_{res}" for res in RESULT_ORDER]
        book_cols = [f"book_prob_{res}" for res in RESULT_ORDER]
        eg_cols = [f"eg_prob_{res}" for res in RESULT_ORDER]
        return pd.DataFrame(columns=base_cols + prob_cols + eg_cols + book_cols + ["bookmaker_pick"])
    df = pd.DataFrame(flat_records)
    df.sort_values(["season", "team", "match_date"], inplace=True)
    return df.reset_index(drop=True)


def _one_hot(actual: pd.Series) -> pd.DataFrame:
    oh = pd.get_dummies(actual)
    for res in RESULT_ORDER:
        if res not in oh:
            oh[res] = 0.0
    return oh[RESULT_ORDER].astype(float)


def accuracy(series: pd.Series, actual: pd.Series) -> float:
    aligned = series.dropna()
    actual_aligned = actual.loc[aligned.index]
    if aligned.empty:
        return float("nan")
    return (aligned == actual_aligned).mean()


def brier_score(probabilities: pd.DataFrame, actual: pd.Series) -> float:
    mask = probabilities.notna().all(axis=1) & actual.notna()
    if mask.sum() == 0:
        return float("nan")
    probs = probabilities.loc[mask, RESULT_ORDER].astype(float).to_numpy()
    actual_one_hot = _one_hot(actual.loc[mask])
    diff = probs - actual_one_hot.to_numpy()
    squared = np.square(diff)
    return float(squared.sum(axis=1).mean())


def evaluate_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-season evaluation metrics for the model and bookmaker.
    """
    results = []
    for season, season_df in predictions_df.groupby("season"):
        model_mask = season_df["predicted"].notna() & season_df["actual"].notna()
        book_mask = season_df["bookmaker_pick"].notna() & season_df["actual"].notna()

        if model_mask.sum():
            season_model_accuracy = accuracy(
                season_df.loc[model_mask, "predicted"],
                season_df.loc[model_mask, "actual"],
            )
            model_probs = season_df.loc[model_mask, [f"model_prob_{r}" for r in RESULT_ORDER]]
            prepared_probs = (
                model_probs.rename(columns=lambda c: c.replace("model_prob_", ""))
                .reindex(columns=RESULT_ORDER)
            )
            season_model_brier = brier_score(
                prepared_probs,
                season_df.loc[model_mask, "actual"],
            )
        else:
            season_model_accuracy = float("nan")
            season_model_brier = float("nan")

        if book_mask.sum():
            season_book_accuracy = accuracy(
                season_df.loc[book_mask, "bookmaker_pick"],
                season_df.loc[book_mask, "actual"],
            )
            book_probs = season_df.loc[book_mask, [f"book_prob_{r}" for r in RESULT_ORDER]]
            prepared_book_probs = (
                book_probs.rename(columns=lambda c: c.replace("book_prob_", ""))
                .reindex(columns=RESULT_ORDER)
            )
            season_book_brier = brier_score(
                prepared_book_probs,
                season_df.loc[book_mask, "actual"],
            )
        else:
            season_book_accuracy = float("nan")
            season_book_brier = float("nan")

        results.append(
            {
                "season": season,
                "model_accuracy": season_model_accuracy,
                "bookmaker_accuracy": season_book_accuracy,
                "model_brier_score": season_model_brier,
                "bookmaker_brier_score": season_book_brier,
            }
        )
    if not results:
        return pd.DataFrame(
            columns=[
                "season",
                "model_accuracy",
                "bookmaker_accuracy",
                "model_brier_score",
                "bookmaker_brier_score",
            ]
        )
    return pd.DataFrame(results)
