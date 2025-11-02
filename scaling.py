"""
Seasonal feature scaling helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


ROBUST_PATTERNS = (
    "shots",
    "fouls",
    "corners",
    "cards",
    "discipline",
)

STANDARD_PATTERNS = (
    "attack_strength",
    "defence_weakness",
    "strength_index",
    "elo",
    "xg_diff",
    "xg_ratio",
    "xg_trend",
    "points_avg",
    "goal_diff",
    "conversion",
    "overperformance",
)


def _matches_pattern(column: str, patterns: Sequence[str]) -> bool:
    return any(pattern in column for pattern in patterns)


@dataclass
class ColumnScaler:
    scaler: object
    columns: Sequence[str] = field(default_factory=list)


@dataclass
class SeasonalScaler:
    """
    Applies per-season scaling with fallbacks to a global scaler.
    """

    robust_columns: Sequence[str]
    standard_columns: Sequence[str]
    robust_scalers: Dict[str, Dict[str, RobustScaler]] = field(default_factory=dict)
    standard_scalers: Dict[str, Dict[str, StandardScaler]] = field(default_factory=dict)
    global_robust: Dict[str, RobustScaler] = field(default_factory=dict)
    global_standard: Dict[str, StandardScaler] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, seasons: Iterable[str]) -> None:
        seasons = pd.Series(list(seasons), index=X.index, dtype=str)
        train_df = X.copy()

        for col in self.robust_columns:
            if col not in train_df.columns:
                continue
            scaler = RobustScaler()
            scaler.fit(train_df[[col]])
            self.global_robust[col] = scaler

        for col in self.standard_columns:
            if col not in train_df.columns:
                continue
            scaler = StandardScaler()
            scaler.fit(train_df[[col]])
            self.global_standard[col] = scaler

        for season in seasons.unique():
            mask = seasons == season
            if not mask.any():
                continue
            season_df = train_df.loc[mask]
            robust_map: Dict[str, RobustScaler] = {}
            standard_map: Dict[str, StandardScaler] = {}
            for col in self.robust_columns:
                if col not in season_df.columns:
                    continue
                scaler = RobustScaler()
                scaler.fit(season_df[[col]])
                robust_map[col] = scaler
            for col in self.standard_columns:
                if col not in season_df.columns:
                    continue
                scaler = StandardScaler()
                scaler.fit(season_df[[col]])
                standard_map[col] = scaler
            self.robust_scalers[season] = robust_map
            self.standard_scalers[season] = standard_map

    def transform(self, X: pd.DataFrame, seasons: Iterable[str]) -> pd.DataFrame:
        seasons = pd.Series(list(seasons), index=X.index, dtype=str)
        transformed = X.copy()

        for col in self.robust_columns:
            if col not in transformed.columns:
                continue
            for season, mask in seasons.groupby(seasons).groups.items():
                scaler = self.robust_scalers.get(season, {}).get(
                    col, self.global_robust.get(col)
                )
                if scaler is None:
                    continue
                vals = transformed.loc[mask, [col]]
                transformed.loc[mask, col] = scaler.transform(vals).astype(float).ravel()

        for col in self.standard_columns:
            if col not in transformed.columns:
                continue
            for season, mask in seasons.groupby(seasons).groups.items():
                scaler = self.standard_scalers.get(season, {}).get(
                    col, self.global_standard.get(col)
                )
                if scaler is None:
                    continue
                vals = transformed.loc[mask, [col]]
                transformed.loc[mask, col] = scaler.transform(vals).astype(float).ravel()

        return transformed

    def transform_array(self, X: pd.DataFrame, seasons: Iterable[str]) -> np.ndarray:
        return self.transform(X, seasons).to_numpy(dtype=float)

    def get_state(self) -> Dict[str, Dict[str, np.ndarray]]:
        def _extract(scaler: Optional[object]) -> Optional[Dict[str, np.ndarray]]:
            if scaler is None:
                return None
            return {
                "center_": getattr(scaler, "center_", None),
                "scale_": getattr(scaler, "scale_", None),
                "quantiles_": getattr(scaler, "quantiles_", None),
            }

        state: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
            "robust_scalers": {},
            "standard_scalers": {},
            "global_robust": {},
            "global_standard": {},
        }
        for season, scalers in self.robust_scalers.items():
            state["robust_scalers"][season] = {
                col: _extract(scaler) for col, scaler in scalers.items()
            }
        for season, scalers in self.standard_scalers.items():
            state["standard_scalers"][season] = {
                col: _extract(scaler) for col, scaler in scalers.items()
            }
        state["global_robust"] = {col: _extract(s) for col, s in self.global_robust.items()}
        state["global_standard"] = {
            col: _extract(s) for col, s in self.global_standard.items()
        }
        return state


def infer_scaler_columns(columns: Iterable[str]) -> Dict[str, Sequence[str]]:
    robust_cols = []
    standard_cols = []
    for col in columns:
        if _matches_pattern(col, ROBUST_PATTERNS):
            robust_cols.append(col)
        elif _matches_pattern(col, STANDARD_PATTERNS):
            standard_cols.append(col)
    return {"robust": sorted(set(robust_cols)), "standard": sorted(set(standard_cols))}
