"""
Online rating updater producing fixture-level probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .models.goal_poisson import lambdas_to_probas


@dataclass
class TeamRatings:
    attack: float = 0.0
    defence: float = 0.0


def _initial_ratings() -> TeamRatings:
    return TeamRatings()


def generate_online_probabilities(
    fixtures: pd.DataFrame,
    offline_lambdas: Dict[str, Tuple[float, float]] | None = None,
    *,
    eta: float = 0.01,
    decay: float = 0.995,
    residual_smoothing: float = 0.6,
    base_rate: float = 1.25,
    home_advantage: float = 0.15,
    max_delta: float = 0.25,
) -> pd.DataFrame:
    """
    Sequentially update team ratings to derive online probabilities.

    Parameters
    ----------
    fixtures : pd.DataFrame
        Fixture-level dataset (one row per match, home perspective).
    offline_lambdas : Optional mapping of match_id -> (lambda_home, lambda_away)
        Offline expected goals used to warm-start the optimiser.
    eta : float
        Learning rate for attack/defence adjustments (log-scale).
    decay : float
        Rating decay applied after each update to avoid runaway values.
    residual_smoothing : float
        EWMA smoothing factor for residuals (0 = no smoothing).
    base_rate : float
        Baseline scoring rate used when offline lambdas are unavailable.
    home_advantage : float
        Additive advantage (log-scale) awarded to the home side.
    max_delta : float
        Cap on per-match attack/defence updates (log units).
    """

    ordered = fixtures.sort_values(["season", "match_date", "home_team"]).copy()
    records = []

    if offline_lambdas is None:
        offline_lookup: Dict[str, Tuple[float, float]] = {}
    else:
        offline_lookup = offline_lambdas
        lambdas = np.array(
            [vals for vals in offline_lookup.values() if vals[0] > 0 and vals[1] > 0]
        )
        if lambdas.size:
            base_rate = float(np.clip(np.mean(lambdas), 0.5, 2.5))

    ratings: Dict[str, Dict[str, TeamRatings]] = {}
    attack_prior: Dict[str, float] = {}
    defence_prior: Dict[str, float] = {}
    residual_attack: Dict[str, float] = {}
    residual_defence: Dict[str, float] = {}

    if offline_lambdas:
        home_means: Dict[str, float] = {}
        away_means: Dict[str, float] = {}
        for _, row in ordered.iterrows():
            match_id = row["match_id"]
            lambdas_pair = offline_lookup.get(match_id)
            if not lambdas_pair:
                continue
            home_mean = home_means.setdefault(row["home_team"], [])
            home_mean.append(lambdas_pair[0])
            away_mean = away_means.setdefault(row["away_team"], [])
            away_mean.append(lambdas_pair[1])
        for team, values in home_means.items():
            mean_lambda = max(float(np.mean(values)), 0.05)
            attack_prior[team] = float(np.log(mean_lambda / base_rate))
        for team, values in away_means.items():
            mean_lambda = max(float(np.mean(values)), 0.05)
            defence_prior[team] = float(np.log(base_rate / mean_lambda))

    def _get_ratings(season: str, team: str) -> TeamRatings:
        season_ratings = ratings.setdefault(season, {})
        rating = season_ratings.get(team)
        if rating is None:
            rating = TeamRatings(
                attack=attack_prior.get(team, 0.0),
                defence=defence_prior.get(team, 0.0),
            )
            season_ratings[team] = rating
        return rating

    for _, row in ordered.iterrows():
        season = row["season"]
        fixture_id = row["match_id"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_rating = _get_ratings(season, home_team)
        away_rating = _get_ratings(season, away_team)

        base_home, base_away = offline_lookup.get(
            fixture_id, (base_rate, base_rate)
        )
        base_home = max(base_home, 0.05)
        base_away = max(base_away, 0.05)

        lambda_home = base_home * float(
            np.exp(home_advantage + home_rating.attack - away_rating.defence)
        )
        lambda_away = base_away * float(
            np.exp(away_rating.attack - home_rating.defence)
        )
        lambda_home = max(lambda_home, 0.05)
        lambda_away = max(lambda_away, 0.05)

        probs = lambdas_to_probas(np.array([lambda_home]), np.array([lambda_away]))
        records.append(
            {
                "match_id": fixture_id,
                "p_online_home": float(probs[0, 0]),
                "p_online_draw": float(probs[0, 1]),
                "p_online_away": float(probs[0, 2]),
            }
        )

        if pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]):
            observed_home = (
                float(row["home_xg"]) if pd.notna(row["home_xg"]) else float(row["home_goals"])
            )
            observed_away = (
                float(row["away_xg"]) if pd.notna(row["away_xg"]) else float(row["away_goals"])
            )

            residual_h = observed_home - lambda_home
            residual_a = observed_away - lambda_away

            prev_res_h = residual_attack.get(home_team, 0.0)
            prev_res_a = residual_attack.get(away_team, 0.0)
            prev_res_def_h = residual_defence.get(home_team, 0.0)
            prev_res_def_a = residual_defence.get(away_team, 0.0)

            smoothed_home_att = (residual_smoothing * prev_res_h) + (
                (1.0 - residual_smoothing) * residual_h
            )
            smoothed_away_att = (residual_smoothing * prev_res_a) + (
                (1.0 - residual_smoothing) * residual_a
            )
            smoothed_home_def = (residual_smoothing * prev_res_def_h) + (
                (1.0 - residual_smoothing) * (-residual_a)
            )
            smoothed_away_def = (residual_smoothing * prev_res_def_a) + (
                (1.0 - residual_smoothing) * (-residual_h)
            )

            residual_attack[home_team] = smoothed_home_att
            residual_attack[away_team] = smoothed_away_att
            residual_defence[home_team] = smoothed_home_def
            residual_defence[away_team] = smoothed_away_def

            delta_attack_home = float(np.clip(eta * smoothed_home_att, -max_delta, max_delta))
            delta_defence_home = float(np.clip(eta * smoothed_home_def, -max_delta, max_delta))
            delta_attack_away = float(np.clip(eta * smoothed_away_att, -max_delta, max_delta))
            delta_defence_away = float(np.clip(eta * smoothed_away_def, -max_delta, max_delta))

            home_rating.attack = float(np.clip(home_rating.attack + delta_attack_home, -1.5, 1.5))
            home_rating.defence = float(np.clip(home_rating.defence + delta_defence_home, -1.5, 1.5))
            away_rating.attack = float(np.clip(away_rating.attack + delta_attack_away, -1.5, 1.5))
            away_rating.defence = float(np.clip(away_rating.defence + delta_defence_away, -1.5, 1.5))

            home_rating.attack *= decay
            home_rating.defence *= decay
            away_rating.attack *= decay
            away_rating.defence *= decay

    return pd.DataFrame.from_records(records)
