"""
Transform raw match-level Premier League CSVs into a team-centric dataset with
rolling features suitable for the online weight-adjusting model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


HOME_PREFIX = "H"
AWAY_PREFIX = "A"

FEATURE_BASE_COLUMNS = [
    ("goals_for", "FTHG", "FTAG"),
    ("goals_against", "FTAG", "FTHG"),
    ("shots_for", "HS", "AS"),
    ("shots_against", "AS", "HS"),
    ("shots_on_target_for", "HST", "AST"),
    ("shots_on_target_against", "AST", "HST"),
    ("corners_for", "HC", "AC"),
    ("corners_against", "AC", "HC"),
    ("fouls_for", "HF", "AF"),
    ("fouls_against", "AF", "HF"),
    ("yellow_cards", "HY", "AY"),
    ("red_cards", "HR", "AR"),
]

DERIVED_FEATURES = {
    "goal_difference": lambda df: df["goals_for"] - df["goals_against"],
    "shot_accuracy_for": lambda df: safe_divide(df["shots_on_target_for"], df["shots_for"]),
    "shot_accuracy_against": lambda df: safe_divide(df["shots_on_target_against"], df["shots_against"]),
    "discipline_index": lambda df: df["yellow_cards"] + 2 * df["red_cards"],
}

ROLLING_WINDOW = 5
EXTENDED_ROLLING_WINDOW = 10


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.0)


def load_raw_season_csv(path: Path, season_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"Raw dataset {path} is missing 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["season"] = season_label
    return df


def expand_matches_to_team_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per team per match with mirrored statistics."""
    home_rows = []
    away_rows = []

    for _, row in raw_df.iterrows():
        home_result = map_result_to_class(row.get("FTR"), home=True)
        away_result = map_result_to_class(row.get("FTR"), home=False)
        match_id = f"{row['season']}_{row.get('Date')}_{row.get('HomeTeam')}_{row.get('AwayTeam')}"
        base_home = {
            "season": row["season"],
            "match_date": row["Date"],
            "team": row["HomeTeam"],
            "opponent": row["AwayTeam"],
            "venue": "home",
            "result": home_result,
            "result_class": home_result,
            "match_id": match_id,
        }
        base_away = {
            "season": row["season"],
            "match_date": row["Date"],
            "team": row["AwayTeam"],
            "opponent": row["HomeTeam"],
            "venue": "away",
            "result": away_result,
            "result_class": away_result,
            "match_id": match_id,
        }

        for feature_name, home_col, away_col in FEATURE_BASE_COLUMNS:
            base_home[feature_name] = row.get(home_col, np.nan)
            base_away[feature_name] = row.get(away_col, np.nan)

        odds_mapping = extract_odds(row)
        for key, value in odds_mapping["home"].items():
            base_home[key] = value
        for key, value in odds_mapping["away"].items():
            base_away[key] = value

        home_rows.append(base_home)
        away_rows.append(base_away)

    team_df = pd.DataFrame(home_rows + away_rows)
    team_df.sort_values(["season", "team", "match_date"], inplace=True)
    team_df.reset_index(drop=True, inplace=True)
    return team_df


def extract_odds(row: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Convert raw bookmaker odds into team-centric columns.
    Returned dict shape: {"home": {"odds_win": ..., ...}, "away": {...}}
    """
    bookmakers = [
        ("B365", "B365H", "B365D", "B365A"),
        ("BW", "BWH", "BWD", "BWA"),
        ("PS", "PSH", "PSD", "PSA"),
        ("WH", "WHH", "WHD", "WHA"),
    ]
    home_odds: Dict[str, float] = {}
    away_odds: Dict[str, float] = {}

    # Prefer Bet365, but fall back to any available bookmaker
    for prefix, home_key, draw_key, away_key in bookmakers:
        h_val = row.get(home_key)
        d_val = row.get(draw_key)
        a_val = row.get(away_key)
        if pd.isna(h_val) or pd.isna(d_val) or pd.isna(a_val):
            continue
        home_odds.update(
            {
                "odds_win": h_val,
                "odds_draw": d_val,
                "odds_loss": a_val,
            }
        )
        away_odds.update(
            {
                "odds_win": a_val,
                "odds_draw": d_val,
                "odds_loss": h_val,
            }
        )
        break

    return {"home": home_odds, "away": away_odds}


def map_result_to_class(ftr_value: str, home: bool) -> str:
    if ftr_value is None:
        return None
    value = str(ftr_value).strip().upper()
    if not value:
        return None
    if home:
        mapping = {"H": "win", "D": "draw", "A": "loss"}
    else:
        mapping = {"H": "loss", "D": "draw", "A": "win"}
    return mapping.get(value)


def add_derived_features(team_df: pd.DataFrame) -> pd.DataFrame:
    for name, func in DERIVED_FEATURES.items():
        team_df[name] = func(team_df)
    return team_df


def compute_rolling_features(team_df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    roll_cols = list(FEATURE_BASE_COLUMNS[i][0] for i in range(len(FEATURE_BASE_COLUMNS)))
    roll_cols += list(DERIVED_FEATURES.keys())

    # Pre-match rolling averages using previous matches only
    for col in roll_cols:
        team_df[f"rolling_{col}"] = (
            team_df.groupby(["season", "team"])[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=[0, 1], drop=True)
        )
    return team_df


def finalise_features(team_df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [col for col in team_df.columns if col.startswith("rolling_")]
    team_df = team_df.copy()
    team_df[feature_columns] = team_df[feature_columns].fillna(0.0)
    return team_df


def add_opponent_features(team_df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [col for col in team_df.columns if col.startswith("rolling_")]
    opponent_df = team_df[["match_id", "team"] + feature_columns].copy()
    opponent_df = opponent_df.rename(columns={"team": "opponent"})
    rename_map = {col: f"opponent_{col}" for col in feature_columns}
    opponent_df = opponent_df.rename(columns=rename_map)

    merged = team_df.merge(opponent_df, on=["match_id", "opponent"], how="left")
    opponent_cols = list(rename_map.values())
    merged[opponent_cols] = merged[opponent_cols].fillna(0.0)
    return merged


def add_differential_features(team_df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [col for col in team_df.columns if col.startswith(("rolling_", "context_"))]
    for col in feature_columns:
        opp_col = f"opponent_{col}"
        if opp_col in team_df.columns:
            team_df[f"diff_{col}"] = team_df[col].astype(float) - team_df[opp_col].astype(float)
    diff_cols = [col for col in team_df.columns if col.startswith("diff_")]
    if diff_cols:
        team_df[diff_cols] = team_df[diff_cols].fillna(0.0)
    return team_df


def _compute_streak(series: pd.Series, positive_values: set[str]) -> list[float]:
    streak: list[float] = []
    count = 0
    for value in series:
        streak.append(float(count))
        if value in positive_values:
            count += 1
        else:
            count = 0
    return streak


def add_context_features(team_df: pd.DataFrame) -> pd.DataFrame:
    team_df = team_df.copy()

    group = team_df.groupby(["season", "team"])

    rest_days = group["match_date"].diff().dt.days
    team_df["context_rest_days"] = rest_days.fillna(7).clip(lower=0, upper=14)

    points_map = {"win": 3, "draw": 1, "loss": 0}
    team_df["_points_for_form"] = team_df["result_class"].map(points_map).fillna(0)

    team_df["context_points_avg_5"] = group["_points_for_form"].transform(
        lambda s: s.shift().rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    team_df["context_points_avg_10"] = group["_points_for_form"].transform(
        lambda s: s.shift().rolling(window=EXTENDED_ROLLING_WINDOW, min_periods=1).mean()
    )

    team_df["context_goal_diff_avg_5"] = group["goal_difference"].transform(
        lambda s: s.fillna(0).shift().rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    team_df["context_goal_diff_avg_10"] = group["goal_difference"].transform(
        lambda s: s.fillna(0).shift().rolling(window=EXTENDED_ROLLING_WINDOW, min_periods=1).mean()
    )

    team_df["context_win_streak"] = group["result_class"].transform(
        lambda s: _compute_streak(s, {"win"})
    )
    team_df["context_unbeaten_streak"] = group["result_class"].transform(
        lambda s: _compute_streak(s, {"win", "draw"})
    )

    context_cols = [col for col in team_df.columns if col.startswith("context_")]
    if context_cols:
        team_df[context_cols] = team_df[context_cols].fillna(0.0)

    team_df = team_df.drop(columns=["_points_for_form"], errors="ignore")

    return team_df


def add_standardised_features(team_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_scale = [
        col
        for col in team_df.columns
        if col.startswith("rolling_") or col.startswith("diff_") or col.startswith("context_")
    ]
    if not columns_to_scale:
        return team_df

    for col in columns_to_scale:
        team_df[f"feat_{col}"] = 0.0

    group_indices = team_df.groupby("season").groups
    for season, indices in group_indices.items():
        subset = team_df.loc[indices, columns_to_scale]
        means = subset.mean()
        stds = subset.std().replace(0, 1.0)
        scaled = (subset - means) / stds
        scaled = scaled.fillna(0.0)
        for col in columns_to_scale:
            team_df.loc[indices, f"feat_{col}"] = scaled[col].to_numpy()

    feature_cols = [f"feat_{col}" for col in columns_to_scale]
    team_df[feature_cols] = team_df[feature_cols].fillna(0.0)

    # Integrity checks: warn on constant columns
    import warnings

    for col in feature_cols:
        col_std = team_df[col].std()
        if pd.isna(col_std) or col_std < 1e-9:
            warnings.warn(
                f"Standardised feature '{col}' exhibits near-zero variance; verify data preprocessing.",
                RuntimeWarning,
            )

    return team_df


def build_dataset(raw_files: Dict[str, Path]) -> pd.DataFrame:
    season_frames = []
    for season_label, csv_path in raw_files.items():
        raw_df = load_raw_season_csv(csv_path, season_label)
        team_df = expand_matches_to_team_rows(raw_df)
        team_df = add_derived_features(team_df)
        team_df = compute_rolling_features(team_df)
        team_df = finalise_features(team_df)
        team_df = add_context_features(team_df)
        team_df = add_opponent_features(team_df)
        team_df = add_differential_features(team_df)
        team_df = add_standardised_features(team_df)
        season_frames.append(team_df)

    dataset = pd.concat(season_frames, ignore_index=True)
    dataset.sort_values(["season", "team", "match_date"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    dataset["team_match_number"] = (
        dataset.groupby(["season", "team"]).cumcount() + 1
    )
    return dataset


def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
