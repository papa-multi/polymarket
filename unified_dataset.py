"""
Build a unified, team-level Premier League dataset with aligned stats and xG.

Only football-data.co.uk (match statistics) and Understat (xG) are used, and the
result covers shared seasons (currently 2021-2022 onward). Team names are
normalised to a single canonical form and the final output contains one record
per team per match with consistent goals/xG values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from football_weight_model.goal_model import attach_expected_goal_features
from football_weight_model.team_names import canonical_team_name


UNDERSTAT_SEASON_MAP = {
    "2021": "2021-2022",
    "2022": "2022-2023",
    "2023": "2023-2024",
    "2024": "2024-2025",
    "2025": "2025-2026",
}

VALID_SEASONS: Iterable[str] = UNDERSTAT_SEASON_MAP.values()


def _normalise_team(name: str) -> str:
    return canonical_team_name(name)


def _safe_float(value: object) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: object) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(int(value))
    except (TypeError, ValueError):
        return float("nan")


def _safe_result(team_goals: pd.Series, opponent_goals: pd.Series) -> pd.Series:
    def decide(pair: tuple[float, float]) -> str:
        a, b = pair
        if pd.isna(a) or pd.isna(b):
            return pd.NA
        if a > b:
            return "win"
        if a < b:
            return "loss"
        return "draw"

    return pd.Series(map(decide, zip(team_goals, opponent_goals)), index=team_goals.index)


def load_football_processed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed football-data dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=["match_date"])
    df = df[df["season"].isin(VALID_SEASONS)].copy()
    if df.empty:
        raise ValueError("No football-data records available for the configured seasons.")

    df["match_date"] = df["match_date"].dt.normalize()
    df["team"] = df["team"].map(_normalise_team)
    df["opponent"] = df["opponent"].map(_normalise_team)

    if pd.isna(df["team"]).any() or pd.isna(df["opponent"]).any():
        raise ValueError("Some team names could not be normalised in football-data dataset.")

    df["result"] = df.get("result_class", _safe_result(df["goals_for"], df["goals_against"]))
    df = df.rename(
        columns={
            "goals_for": "team_goals",
            "goals_against": "opponent_goals",
        }
    )
    return df


def load_understat_team_rows(raw_dir: Path) -> pd.DataFrame:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Understat directory not found: {raw_dir}")

    rows: List[dict] = []
    for path in sorted(raw_dir.glob("matches_EPL_*.json")):
        season_code = path.stem.split("_")[-1]
        season_label = UNDERSTAT_SEASON_MAP.get(season_code)
        if season_label is None:
            continue
        data = json.loads(path.read_text())
        for match in data:
            match_id = int(match["id"])
            match_date = pd.to_datetime(match["datetime"]).normalize()
            home = _normalise_team(match["h"]["title"])
            away = _normalise_team(match["a"]["title"])
            home_goals = _safe_int(match["goals"].get("h"))
            away_goals = _safe_int(match["goals"].get("a"))
            home_xg = _safe_float(match["xG"].get("h"))
            away_xg = _safe_float(match["xG"].get("a"))
            rows.append(
                {
                    "season": season_label,
                    "match_date": match_date,
                    "understat_match_id": match_id,
                    "team": home,
                    "opponent": away,
                    "venue": "home",
                    "team_goals": home_goals,
                    "opponent_goals": away_goals,
                    "team_xg": home_xg,
                    "opponent_xg": away_xg,
                }
            )
            rows.append(
                {
                    "season": season_label,
                    "match_date": match_date,
                    "understat_match_id": match_id,
                    "team": away,
                    "opponent": home,
                    "venue": "away",
                    "team_goals": away_goals,
                    "opponent_goals": home_goals,
                    "team_xg": away_xg,
                    "opponent_xg": home_xg,
                }
            )

    if not rows:
        raise RuntimeError("No Understat matches were parsed from the provided directory.")
    df = pd.DataFrame(rows)
    return df


def augment_unified_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional rolling, strength, and market features for the unified dataset."""
    required_columns = {"season", "team", "opponent", "match_date", "venue"}
    if not required_columns.issubset(df.columns):
        return df
    already_augmented_markers = {
        "team_strength_index",
        "opponent_strength_index",
        "team_points_before",
        "opponent_points_before",
    }
    if already_augmented_markers.issubset(df.columns):
        return df

    working = df.copy()

    # Ensure bookmaker probabilities exist
    for col in ["book_prob_win", "book_prob_draw", "book_prob_loss"]:
        if col not in working.columns:
            working[col] = np.nan
    odds_cols = ["odds_win", "odds_draw", "odds_loss"]
    if set(odds_cols).issubset(working.columns):
        odds_frame = working[odds_cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = 1.0 / odds_frame.replace(0, np.nan)
        implied_sum = implied.sum(axis=1)
        prob_df = implied.div(implied_sum, axis=0)
        for src, target in zip(odds_cols, ["book_prob_win", "book_prob_draw", "book_prob_loss"]):
            working[target] = working[target].fillna(prob_df[src])
    working[["book_prob_win", "book_prob_draw", "book_prob_loss"]] = working[
        ["book_prob_win", "book_prob_draw", "book_prob_loss"]
    ].fillna(1.0 / 3.0)

    points_map = {"win": 3.0, "draw": 1.0, "loss": 0.0}
    working["team_points"] = working["result"].map(points_map).astype(float)

    team_group = working.groupby(["season", "team"], group_keys=False)
    cumulative_points = team_group["team_points"].cumsum()
    working["team_points_before"] = (cumulative_points - working["team_points"]).fillna(0.0)
    working["team_points_avg_5"] = team_group["team_points"].transform(
        lambda s: s.shift().rolling(window=5, min_periods=1).mean()
    ).fillna(0.0)

    home_points = working["team_points"].where(working["venue"] == "home")
    away_points = working["team_points"].where(working["venue"] == "away")
    home_group = home_points.groupby([working["season"], working["team"]])
    away_group = away_points.groupby([working["season"], working["team"]])
    working["team_points_home_avg_5"] = home_group.transform(
        lambda s: s.shift().rolling(window=5, min_periods=1).mean()
    ).fillna(0.0)
    working["team_points_away_avg_5"] = away_group.transform(
        lambda s: s.shift().rolling(window=5, min_periods=1).mean()
    ).fillna(0.0)

    working["rest_days"] = team_group["match_date"].transform(
        lambda s: s.diff().dt.days.clip(lower=0, upper=14)
    ).fillna(7.0)

    matches_played_prior = team_group.cumcount()
    working["team_strength_index"] = np.divide(
        working["team_points_before"],
        np.maximum(matches_played_prior, 1),
        out=np.zeros_like(working["team_points_before"]),
        where=np.maximum(matches_played_prior, 1) != 0,
    )

    # Elo-style ratings
    if "match_id" not in working.columns:
        working["match_id"] = (
            working["season"].astype(str)
            + "_"
            + working["match_date"].astype(str)
            + "_"
            + working["team"]
            + "_"
            + working["opponent"]
        )

    working = working.reset_index().rename(columns={"index": "__orig_index"})
    working = working.sort_values(["season", "match_date", "match_id", "team"]).reset_index(drop=True)
    working["team_elo_pre"] = 1500.0
    working["opponent_elo_pre"] = 1500.0
    working["elo_expected_score"] = 0.5

    ratings: Dict[Tuple[str, str], float] = {}
    result_to_score = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    K_FACTOR = 24.0
    HOME_ADVANTAGE = 50.0

    def _expected_score(r_team: float, r_opp: float, venue: str) -> float:
        advantage = HOME_ADVANTAGE if venue == "home" else (-HOME_ADVANTAGE if venue == "away" else 0.0)
        exponent = (r_opp - r_team + advantage) / 400.0
        return 1.0 / (1.0 + 10.0 ** exponent)

    for (season, match_id), match_df in working.groupby(["season", "match_id"], sort=False):
        match_rows = match_df.copy()
        for idx, row in match_rows.iterrows():
            team_key = (season, row["team"])
            opponent_key = (season, row["opponent"])
            team_rating = ratings.get(team_key, 1500.0)
            opponent_rating = ratings.get(opponent_key, 1500.0)
            working.at[idx, "team_elo_pre"] = team_rating
            working.at[idx, "opponent_elo_pre"] = opponent_rating
            working.at[idx, "elo_expected_score"] = _expected_score(team_rating, opponent_rating, row["venue"])

        home_rows = match_rows[match_rows["venue"] == "home"]
        away_rows = match_rows[match_rows["venue"] == "away"]
        if home_rows.empty or away_rows.empty:
            match_rows = match_rows.sort_values("venue")
            home_rows = match_rows.iloc[[0]]
            away_rows = match_rows.iloc[[1 if len(match_rows) > 1 else 0]]

        home_idx = home_rows.index[0]
        away_idx = away_rows.index[0]

        home_team = working.loc[home_idx, "team"]
        away_team = working.loc[away_idx, "team"]

        rating_home = working.loc[home_idx, "team_elo_pre"]
        rating_away = working.loc[away_idx, "team_elo_pre"]

        expected_home = _expected_score(rating_home, rating_away, "home")
        actual_home = result_to_score.get(working.loc[home_idx, "result"], 0.5)

        expected_away = 1.0 - expected_home
        actual_away = 1.0 - actual_home

        ratings[(season, home_team)] = rating_home + K_FACTOR * (actual_home - expected_home)
        ratings[(season, away_team)] = rating_away + K_FACTOR * (actual_away - expected_away)

    working["elo_diff"] = working["team_elo_pre"] - working["opponent_elo_pre"]
    working["elo_edge"] = working["elo_expected_score"] - 0.5

    working = working.sort_values("__orig_index").drop(columns=["__orig_index"]).reset_index(drop=True)
    team_group = working.groupby(["season", "team"], group_keys=False)

    working["team_xg_diff_std_5"] = team_group["xg_diff"].transform(
        lambda s: s.shift().rolling(window=5, min_periods=2).std()
    ).fillna(0.0)
    working["matchup_xg_diff_avg_3"] = (
        working.groupby(["team", "opponent"], group_keys=False)["xg_diff"]
        .transform(lambda s: s.shift().rolling(window=3, min_periods=1).mean())
        .fillna(0.0)
    )

    def _rolling_trend(series: pd.Series, window: int) -> pd.Series:
        def slope(x: pd.Series) -> float:
            idx = np.arange(len(x))
            mask = ~np.isnan(x)
            if mask.sum() < 2:
                return 0.0
            return np.polyfit(idx[mask], x[mask], 1)[0]

        return series.shift().rolling(window=window, min_periods=3).apply(slope, raw=False)

    for window in (5, 8):
        working[f"team_points_trend_{window}"] = (
            working.groupby(["season", "team"])["team_points_avg_5"].transform(lambda s: _rolling_trend(s, window))
        )
        working[f"team_xg_trend_{window}"] = (
            working.groupby(["season", "team"])["team_xg"].transform(lambda s: _rolling_trend(s, window))
        )

    eps = 1e-6
    for window in (5, 8):
        goals_sum = working.groupby(["season", "team"])["team_goals"].transform(
            lambda s: s.fillna(0.0).shift().rolling(window=window, min_periods=1).sum()
        )
        xg_sum = working.groupby(["season", "team"])["team_xg"].transform(
            lambda s: s.shift().rolling(window=window, min_periods=1).sum()
        )
        efficiency = goals_sum / np.clip(xg_sum, eps, None)
        overperf = goals_sum - xg_sum
        working[f"team_xg_conversion_{window}"] = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        working[f"team_xg_overperformance_{window}"] = overperf.fillna(0.0)

    opponent_columns = [
        "match_id",
        "team",
        "team_points_before",
        "team_points_avg_5",
        "team_points_home_avg_5",
        "team_points_away_avg_5",
        "team_strength_index",
        "rest_days",
        "team_xg_diff_std_5",
        "book_prob_win",
        "book_prob_draw",
        "book_prob_loss",
        "team_points_trend_5",
        "team_points_trend_8",
        "team_xg_trend_5",
        "team_xg_trend_8",
        "team_xg_conversion_5",
        "team_xg_conversion_8",
        "team_xg_overperformance_5",
        "team_xg_overperformance_8",
    ]
    opponent_frame = working[opponent_columns].rename(
        columns={
            "team": "opponent",
            "team_points_before": "opponent_points_before",
            "team_points_avg_5": "opponent_points_avg_5",
            "team_points_home_avg_5": "opponent_points_home_avg_5",
            "team_points_away_avg_5": "opponent_points_away_avg_5",
            "team_strength_index": "opponent_strength_index",
            "rest_days": "opponent_rest_days",
            "team_xg_diff_std_5": "opponent_xg_diff_std_5",
            "book_prob_win": "opponent_book_prob_win",
            "book_prob_draw": "opponent_book_prob_draw",
            "book_prob_loss": "opponent_book_prob_loss",
            "team_points_trend_5": "opponent_points_trend_5",
            "team_points_trend_8": "opponent_points_trend_8",
            "team_xg_trend_5": "opponent_xg_trend_5",
            "team_xg_trend_8": "opponent_xg_trend_8",
            "team_xg_conversion_5": "opponent_xg_conversion_5",
            "team_xg_conversion_8": "opponent_xg_conversion_8",
            "team_xg_overperformance_5": "opponent_xg_overperformance_5",
            "team_xg_overperformance_8": "opponent_xg_overperformance_8",
        }
    )
    working = working.merge(opponent_frame, on=["match_id", "opponent"], how="left")

    for col, default in [
        ("opponent_points_before", 0.0),
        ("opponent_points_avg_5", 0.0),
        ("opponent_points_home_avg_5", 0.0),
        ("opponent_points_away_avg_5", 0.0),
        ("opponent_strength_index", 0.0),
        ("opponent_rest_days", 7.0),
        ("opponent_xg_diff_std_5", 0.0),
        ("opponent_book_prob_win", 1.0 / 3.0),
        ("opponent_book_prob_draw", 1.0 / 3.0),
        ("opponent_book_prob_loss", 1.0 / 3.0),
        ("opponent_points_trend_5", 0.0),
        ("opponent_points_trend_8", 0.0),
        ("opponent_xg_trend_5", 0.0),
        ("opponent_xg_trend_8", 0.0),
        ("opponent_xg_conversion_5", 1.0),
        ("opponent_xg_conversion_8", 1.0),
        ("opponent_xg_overperformance_5", 0.0),
        ("opponent_xg_overperformance_8", 0.0),
    ]:
        if col in working.columns:
            working[col] = working[col].fillna(default)

    working["strength_index_diff"] = working["team_strength_index"] - working["opponent_strength_index"]
    working["points_avg_5_diff"] = working["team_points_avg_5"] - working["opponent_points_avg_5"]
    working["rest_days_diff"] = working["rest_days"] - working["opponent_rest_days"]
    working["market_confidence_gap"] = working["book_prob_win"] - working["opponent_book_prob_win"]

    if {"team_points_trend_5", "opponent_points_trend_5"}.issubset(working.columns):
        working["points_trend_diff_5"] = (
            working["team_points_trend_5"] - working["opponent_points_trend_5"]
        )
    if {"team_points_trend_8", "opponent_points_trend_8"}.issubset(working.columns):
        working["points_trend_diff_8"] = (
            working["team_points_trend_8"] - working["opponent_points_trend_8"]
        )
    if {"team_xg_trend_5", "opponent_xg_trend_5"}.issubset(working.columns):
        working["xg_trend_diff_5"] = (
            working["team_xg_trend_5"] - working["opponent_xg_trend_5"]
        )
    if {"team_xg_trend_8", "opponent_xg_trend_8"}.issubset(working.columns):
        working["xg_trend_diff_8"] = (
            working["team_xg_trend_8"] - working["opponent_xg_trend_8"]
        )
    if {"team_xg_conversion_5", "opponent_xg_conversion_5"}.issubset(working.columns):
        working["xg_conversion_ratio_diff_5"] = (
            working["team_xg_conversion_5"] - working["opponent_xg_conversion_5"]
        )
    if {"team_xg_conversion_8", "opponent_xg_conversion_8"}.issubset(working.columns):
        working["xg_conversion_ratio_diff_8"] = (
            working["team_xg_conversion_8"] - working["opponent_xg_conversion_8"]
        )
    if {"team_xg_overperformance_5", "opponent_xg_overperformance_5"}.issubset(working.columns):
        working["xg_overperformance_diff_5"] = (
            working["team_xg_overperformance_5"] - working["opponent_xg_overperformance_5"]
        )
    if {"team_xg_overperformance_8", "opponent_xg_overperformance_8"}.issubset(working.columns):
        working["xg_overperformance_diff_8"] = (
            working["team_xg_overperformance_8"] - working["opponent_xg_overperformance_8"]
        )

    working["team_expected_points_market"] = (
        3.0 * working["book_prob_win"] + working["book_prob_draw"]
    )
    working["opponent_expected_points_market"] = (
        3.0 * working["opponent_book_prob_win"] + working["opponent_book_prob_draw"]
    )
    working["expected_points_edge"] = working["team_points_avg_5"] - working["team_expected_points_market"]

    engineered_new = [
        "team_points_before",
        "team_points_avg_5",
        "team_points_home_avg_5",
        "team_points_away_avg_5",
        "opponent_points_before",
        "opponent_points_avg_5",
        "opponent_points_home_avg_5",
        "opponent_points_away_avg_5",
        "team_strength_index",
        "opponent_strength_index",
        "strength_index_diff",
        "points_avg_5_diff",
        "rest_days",
        "opponent_rest_days",
        "rest_days_diff",
        "team_xg_diff_std_5",
        "opponent_xg_diff_std_5",
        "matchup_xg_diff_avg_3",
        "team_expected_points_market",
        "opponent_expected_points_market",
        "expected_points_edge",
        "market_confidence_gap",
        "opponent_book_prob_win",
        "opponent_book_prob_draw",
        "opponent_book_prob_loss",
        "team_xg_conversion_5",
        "team_xg_conversion_8",
        "team_xg_overperformance_5",
        "team_xg_overperformance_8",
        "opponent_xg_conversion_5",
        "opponent_xg_conversion_8",
        "opponent_xg_overperformance_5",
        "opponent_xg_overperformance_8",
        "team_elo_pre",
        "opponent_elo_pre",
        "elo_diff",
        "elo_expected_score",
        "elo_edge",
    ]

    engineered_new.extend(
        [
            "team_points_trend_5",
            "team_points_trend_8",
            "team_xg_trend_5",
            "team_xg_trend_8",
            "opponent_points_trend_5",
            "opponent_points_trend_8",
            "opponent_xg_trend_5",
            "opponent_xg_trend_8",
            "points_trend_diff_5",
            "points_trend_diff_8",
            "xg_trend_diff_5",
            "xg_trend_diff_8",
            "xg_conversion_ratio_diff_5",
            "xg_conversion_ratio_diff_8",
            "xg_overperformance_diff_5",
            "xg_overperformance_diff_8",
        ]
    )

    working = working.drop(columns=["team_points"])
    for col in engineered_new:
        working[col] = working[col].astype(float).fillna(0.0)

    trend_cols = [col for col in working.columns if col.startswith(("team_points_trend_", "team_xg_trend_"))]
    if trend_cols:
        for col in trend_cols:
            working[col] = working[col].fillna(0.0)

    return working


@dataclass
class UnifiedDatasetPaths:
    processed_dataset: Path
    understat_dir: Path
    output_path: Path

    @classmethod
    def default(cls, root: Path) -> "UnifiedDatasetPaths":
        return cls(
            processed_dataset=root / "data" / "processed" / "epl_rolling_features.csv",
            understat_dir=root / "football_weight_model" / "data" / "raw" / "understat",
            output_path=root / "data" / "processed" / "epl_unified_team_matches.csv",
        )


def build_unified_dataset(paths: UnifiedDatasetPaths) -> pd.DataFrame:
    football_df = load_football_processed(paths.processed_dataset)
    understat_df = load_understat_team_rows(paths.understat_dir)

    valid_seasons = set(football_df["season"].unique())
    understat_df = understat_df[understat_df["season"].isin(valid_seasons)].copy()

    merge_cols = ["season", "match_date", "team", "opponent", "venue"]

    # Add placeholder rows for future fixtures that exist in Understat but not football-data
    if not football_df.empty:
        existing_keys = football_df[merge_cols].drop_duplicates()
        missing = (
            understat_df.merge(existing_keys, on=merge_cols, how="left", indicator=True)
            .loc[lambda df: df["_merge"] == "left_only", merge_cols + ["understat_match_id"]]
            .drop_duplicates()
        )

        if not missing.empty:
            placeholder_rows = []
            columns = football_df.columns.tolist()
            for _, row in missing.iterrows():
                placeholder = {col: np.nan for col in columns}
                placeholder.update(
                    {
                        "season": row["season"],
                        "match_date": row["match_date"],
                        "team": row["team"],
                        "opponent": row["opponent"],
                        "venue": row["venue"],
                        "result": pd.NA if "result" in columns else np.nan,
                        "result_class": pd.NA if "result_class" in columns else np.nan,
                    }
                )

                match_id = None
                if "match_id" in columns:
                    match_id = (
                        f"{row['season']}_"
                        f"{row['match_date'].strftime('%Y-%m-%d')}_"
                        f"{row['team']}_"
                        f"{row['opponent']}"
                    )
                    placeholder["match_id"] = match_id

                if "understat_match_id" in columns:
                    placeholder["understat_match_id"] = row["understat_match_id"]

                placeholder_rows.append(placeholder)

            if placeholder_rows:
                placeholders_df = pd.DataFrame(placeholder_rows, columns=columns)
                football_df = pd.concat([football_df, placeholders_df], ignore_index=True)

    understat_subset = understat_df[merge_cols + ["team_xg", "opponent_xg", "team_goals", "opponent_goals"]].rename(
        columns={
            "team_goals": "team_goals_understat",
            "opponent_goals": "opponent_goals_understat",
        }
    )

    merged = football_df.merge(understat_subset, on=merge_cols, how="left")

    missing_xg_mask = merged["team_xg"].isna()
    if missing_xg_mask.any():
        observed_results = merged["team_goals"].notna()
        real_missing = merged[missing_xg_mask & observed_results][["season", "match_date", "team", "opponent"]]
        if not real_missing.empty:
            raise ValueError(f"Missing Understat xG for some matches:\n{real_missing.head()}")

    tg_compare = merged["team_goals"].notna() & merged["team_goals_understat"].notna()
    og_compare = merged["opponent_goals"].notna() & merged["opponent_goals_understat"].notna()
    team_goals_diff = tg_compare & ~np.isclose(
        merged["team_goals"].astype(float), merged["team_goals_understat"].astype(float)
    )
    opponent_goals_diff = og_compare & ~np.isclose(
        merged["opponent_goals"].astype(float), merged["opponent_goals_understat"].astype(float)
    )
    goals_mismatch = merged[team_goals_diff | opponent_goals_diff]
    if not goals_mismatch.empty:
        raise ValueError("Goal totals differ between sources; investigate alignment issues.")

    merged = merged.drop(columns=["team_goals_understat", "opponent_goals_understat"])

    for col in ["team_xg", "opponent_xg"]:
        merged[col] = merged[col].astype(float)

    merged["xg_diff"] = merged["team_xg"] - merged["opponent_xg"]
    merged["xg_ratio"] = merged.apply(
        lambda row: row["team_xg"] / row["opponent_xg"] if row["opponent_xg"] not in (0.0, np.nan) else 0.0,
        axis=1,
    )

    merged = merged.sort_values(["season", "team", "match_date"]).reset_index(drop=True)

    # Pre-match sequential position
    merged["team_match_number"] = merged.groupby(["season", "team"]).cumcount() + 1

    # Rolling and exponential strength indicators (pre-match only)
    def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
        return series.shift().rolling(window=window, min_periods=1).mean()

    for window in (3, 5):
        merged[f"team_xg_rolling_{window}"] = (
            merged.groupby(["season", "team"]) ["team_xg"].transform(lambda s: _rolling_mean(s, window))
        )
        merged[f"opponent_xg_rolling_{window}"] = (
            merged.groupby(["season", "team"]) ["opponent_xg"].transform(lambda s: _rolling_mean(s, window))
        )
        merged[f"xg_diff_rolling_{window}"] = (
            merged[f"team_xg_rolling_{window}"] - merged[f"opponent_xg_rolling_{window}"]
        )

    alpha = 0.2
    merged["attack_strength"] = (
        merged.groupby(["season", "team"]) ["team_xg"]
        .transform(lambda s: s.shift().ewm(alpha=alpha, adjust=False).mean())
    )
    merged["defence_weakness"] = (
        merged.groupby(["season", "team"]) ["opponent_xg"]
        .transform(lambda s: s.shift().ewm(alpha=alpha, adjust=False).mean())
    )
    merged["attack_vs_defence"] = merged["attack_strength"] - merged["defence_weakness"]
    merged["xg_form_mismatch"] = merged["team_xg_rolling_3"] - merged["opponent_xg_rolling_3"]

    engineered_cols = [
        "team_match_number",
        "team_xg_rolling_3",
        "team_xg_rolling_5",
        "opponent_xg_rolling_3",
        "opponent_xg_rolling_5",
        "xg_diff_rolling_3",
        "xg_diff_rolling_5",
        "attack_strength",
        "defence_weakness",
        "attack_vs_defence",
        "xg_form_mismatch",
    ]

    for col in engineered_cols:
        merged[col] = merged[col].fillna(0.0)

    merged = augment_unified_features(merged)
    merged = attach_expected_goal_features(merged)

    return merged


def save_unified_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
    except PermissionError:
        pass


def main(paths: UnifiedDatasetPaths | None = None) -> Path:
    from .fixture_features import build_unified_dataset as build_fixture_unified_dataset

    result = build_fixture_unified_dataset()
    return result.dataset_path


if __name__ == "__main__":  # pragma: no cover
    output = main()
    print(f"Unified dataset written to {output}")
