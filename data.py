"""
Utility helpers for loading and preparing football match datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from .unified_dataset import augment_unified_features


@dataclass(frozen=True)
class DatasetConfig:
    season_column: str = "season"
    date_column: str = "match_date"
    team_column: str = "team"
    opponent_column: str = "opponent"
    result_column: str = "result"
    odds_columns: Sequence[str] = ("odds_win", "odds_draw", "odds_loss")

    def bookmaker_probability_column_names(self) -> List[str]:
        return ["book_prob_win", "book_prob_draw", "book_prob_loss"]


def _normalise_result(value: str) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().lower()
    mapping = {
        "w": "win",
        "win": "win",
        "home": "win",
        "h": "win",
        "d": "draw",
        "draw": "draw",
        "t": "draw",
        "l": "loss",
        "loss": "loss",
        "away": "loss",
        "a": "loss",
    }
    return mapping.get(value, None)


def load_matches_csv(
    path: Path,
    dataset_config: DatasetConfig,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Load a CSV file containing match level data. The file should already be in
    team-long format (each row describes a match from a single team's
    perspective). Columns listed in `feature_columns` must be numeric.
    """
    data = pd.read_csv(path, parse_dates=[dataset_config.date_column])
    data = augment_unified_features(data)

    required_cols = {
        dataset_config.season_column,
        dataset_config.date_column,
        dataset_config.team_column,
        dataset_config.opponent_column,
        dataset_config.result_column,
    }.union(feature_columns)

    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Make sure we can compute bookmaker probabilities if the odds columns exist
    odds_cols = dataset_config.odds_columns
    if not set(odds_cols).issubset(data.columns):
        odds_cols = []

    data = data.copy()
    data["result_class"] = data[dataset_config.result_column].map(_normalise_result)

    invalid_mask = data[dataset_config.result_column].notna() & data["result_class"].isna()
    if invalid_mask.any():
        raise ValueError("Unable to normalise some result values into win/draw/loss.")

    data = (
        data
        .sort_values([dataset_config.team_column, dataset_config.date_column])
        .reset_index(drop=True)
    )

    data["team_match_number"] = (
        data.groupby(
            [dataset_config.season_column, dataset_config.team_column]
        ).cumcount() + 1
    )

    if odds_cols:
        implied = 1 / data.loc[:, odds_cols]
        implied_sum = implied.sum(axis=1)
        probabilities = implied.div(implied_sum, axis=0)
        book_cols = dataset_config.bookmaker_probability_column_names()
        data.loc[:, book_cols] = probabilities.to_numpy()

    return data
