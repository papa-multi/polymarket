"""
Orchestrates the end-to-end workflow described in the user prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .data import DatasetConfig, load_matches_csv
from .evaluation import evaluate_predictions, predictions_to_dataframe
from .model import OnlineOutcomeModel


@dataclass
class PipelineConfig:
    dataset_path: Path
    dataset_config: DatasetConfig
    feature_columns: Sequence[str]
    seasons: Sequence[str]
    learning_rate: float = 0.1
    warmup_matches: int = 5
    baseline_prob_columns: Optional[Dict[str, str]] = None


def run_pipeline(config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    matches = load_matches_csv(
        config.dataset_path,
        dataset_config=config.dataset_config,
        feature_columns=config.feature_columns,
    )

    matches = matches[matches[config.dataset_config.season_column].isin(config.seasons)]
    if matches.empty:
        raise ValueError("No matches found for the requested seasons.")

    model = OnlineOutcomeModel(
        feature_names=config.feature_columns,
        learning_rate=config.learning_rate,
        baseline_prob_columns=config.baseline_prob_columns or {},
        warmup_matches=config.warmup_matches,
    )

    all_predictions = []

    for season in config.seasons:
        season_matches = matches[matches[config.dataset_config.season_column] == season]
        if season_matches.empty:
            continue
        preds = model.run_season(
            season=season,
            matches=season_matches,
            warmup_matches=config.warmup_matches,
        )
        all_predictions.extend(preds)

    predictions_df = predictions_to_dataframe(all_predictions)
    metrics_df = evaluate_predictions(predictions_df)

    overall_accuracy = predictions_df.loc[
        predictions_df["predicted"].notna() & predictions_df["actual"].notna(),
        "correct",
    ].mean()
    bookmaker_accuracy = None
    if "bookmaker_pick" in predictions_df.columns:
        mask = predictions_df["bookmaker_pick"].notna() & predictions_df["actual"].notna()
        if mask.sum():
            bookmaker_accuracy = (
                predictions_df.loc[mask, "bookmaker_pick"] == predictions_df.loc[mask, "actual"]
            ).mean()

    summary = pd.DataFrame(
        [
            {
                "metric": "model_overall_accuracy",
                "value": overall_accuracy,
            },
            {
                "metric": "bookmaker_overall_accuracy",
                "value": bookmaker_accuracy,
            },
        ]
    )

    metrics_df["accuracy_change_from_previous"] = metrics_df["model_accuracy"].diff()

    return {
        "matches": matches,
        "predictions": predictions_df,
        "season_metrics": metrics_df,
        "summary": summary,
    }
