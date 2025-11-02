"""
Market implied probabilities for diagnostics (no use in training).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_probabilities(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute implied probabilities from bookmaker odds (after removing overround).

    Parameters
    ----------
    odds_df : pd.DataFrame
        Must contain columns: home_odds_win, home_odds_draw, home_odds_loss.
        (Loss corresponds to away win odds from the bookmaker perspective.)
    """
    probs = odds_df.copy()
    for col in ["home_odds_win", "home_odds_draw", "home_odds_loss"]:
        probs[col] = probs[col].replace(0.0, np.nan)

    inv = 1.0 / probs[["home_odds_win", "home_odds_draw", "home_odds_loss"]]
    inv = inv.replace([np.inf, -np.inf], np.nan)
    overround = inv.sum(axis=1)
    implied = inv.div(overround, axis=0)
    implied.columns = ["p_home_market", "p_draw_market", "p_away_market"]
    return implied


def compare_model_vs_market(
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    odds = forecast_df[["home_odds_win", "home_odds_draw", "home_odds_loss"]]
    implied = implied_probabilities(odds)
    comparison = forecast_df[
        ["match_date", "home_team", "away_team", "p_home_win", "p_draw", "p_away_win"]
    ].copy()
    comparison = pd.concat([comparison, implied], axis=1)
    for model_col, market_col, gap_name in [
        ("p_home_win", "p_home_market", "gap_home"),
        ("p_draw", "p_draw_market", "gap_draw"),
        ("p_away_win", "p_away_market", "gap_away"),
    ]:
        comparison[gap_name] = comparison[model_col] - comparison[market_col]
    return comparison
