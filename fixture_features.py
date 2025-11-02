"""
Fixture-level feature engineering for Premier League matches.

This module transforms the team-level unified dataset into fixture rows (one per
match) and computes pre-match features such as exponentially weighted form,
finishing efficiency, trends, matchup contrasts, and context variables.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .team_names import canonical_team_name
from .workflow import WorkflowPaths, default_config, seasons_from_config, prepare_dataset

EPS = 1e-6
HALFLIFE_SHORT = 3.0
HALFLIFE_LONG = 6.0
TREND_WINDOWS = (5, 8)
HOME_ADVANTAGE = 0.25
STRENGTH_DECAY = 0.05
STRENGTH_K = 0.12
EXCLUDED_FEATURE_COLUMNS = {"feat_xg_diff", "feat_goal_diff"}
UPCOMING_FIXTURES_FILENAME = "upcoming_fixtures.csv"

logger = logging.getLogger(__name__)


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


def _points_from_result(result: str) -> float:
    if pd.isna(result):
        return np.nan
    if result == "win":
        return 3.0
    if result == "draw":
        return 1.0
    if result == "loss":
        return 0.0
    return np.nan


def _ewm(series: pd.Series, halflife: float) -> pd.Series:
    """Exponentially weighted mean on prior observations (leakage-safe)."""
    return series.shift().ewm(halflife=halflife, adjust=False).mean()


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Slope of last `window` observations (leakage-safe, requires >=3 points)."""

    def slope(values: pd.Series) -> float:
        idx = np.arange(len(values))
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return 0.0
        x = idx[mask]
        y = values.to_numpy()[mask]
        # polyfit of degree 1: slope only
        return np.polyfit(x, y, 1)[0]

    return series.shift().rolling(window=window, min_periods=3).apply(slope, raw=False)


def _append_dynamic_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute opponent-weighted dynamic team strength ratings updated after each match.

    Ratings are expressed on the xG differential scale and decay toward zero over time.
    """
    df = df.sort_values(["match_date", "match_id", "venue"]).copy()
    strengths: Dict[str, float] = {}
    strength_pre: Dict[int, float] = {}
    opponent_pre: Dict[int, float] = {}

    for match_id, match_rows in df.groupby("match_id", sort=False):
        if match_rows.shape[0] < 2:
            continue
        home_row = match_rows[match_rows["venue"] == "home"].iloc[0]
        away_row = match_rows[match_rows["venue"] == "away"].iloc[0]

        home_team = str(home_row["team"])
        away_team = str(away_row["team"])

        home_rating = strengths.get(home_team, 0.0)
        away_rating = strengths.get(away_team, 0.0)

        strength_pre[home_row.name] = home_rating
        strength_pre[away_row.name] = away_rating
        opponent_pre[home_row.name] = away_rating
        opponent_pre[away_row.name] = home_rating

        home_xg = _safe_float(home_row.get("team_xg"))
        away_xg = _safe_float(away_row.get("team_xg"))
        if np.isnan(home_xg) or np.isnan(away_xg):
            xg_diff = 0.0
        else:
            xg_diff = home_xg - away_xg

        expected_diff = home_rating - away_rating + HOME_ADVANTAGE
        delta = STRENGTH_K * (xg_diff - expected_diff)

        strengths[home_team] = (1.0 - STRENGTH_DECAY) * (home_rating + delta)
        strengths[away_team] = (1.0 - STRENGTH_DECAY) * (away_rating - delta)

    df["feat_dynamic_strength"] = df.index.to_series().map(strength_pre).fillna(0.0).astype(float)
    df["feat_opponent_strength"] = df.index.to_series().map(opponent_pre).fillna(0.0).astype(float)
    df["feat_strength_edge"] = df["feat_dynamic_strength"] - df["feat_opponent_strength"]
    df["feat_dynamic_strength_ewm"] = (
        df.groupby("team")["feat_dynamic_strength"]
        .transform(lambda s: _ewm(s.fillna(0.0), HALFLIFE_LONG))
        .fillna(0.0)
    )
    return df


def _prepare_team_history(team_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team pre-match features used later at fixture level."""
    df = team_df.sort_values(["team", "match_date"]).copy()

    numeric_columns = [
        "team_xg",
        "opponent_xg",
        "team_goals",
        "opponent_goals",
        "team_shots",
        "opponent_shots",
        "team_shots_on_target",
        "opponent_shots_on_target",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["team_points_raw"] = df["result_class"].map(_points_from_result).astype(float).fillna(0.0)
    df["team_match_number"] = df.groupby("team").cumcount()
    df["rest_days"] = (
        df.groupby("team")["match_date"]
        .diff()
        .dt.days.fillna(7)
        .clip(lower=0, upper=14)
    )
    df["feat_rest_days"] = df["rest_days"].fillna(0.0)

    metric_sources = {
        "xg_for": "team_xg",
        "xg_against": "opponent_xg",
        "goals_for": "team_goals",
        "goals_against": "opponent_goals",
        "shots_for": "team_shots",
        "shots_against": "opponent_shots",
        "shots_on_target_for": "team_shots_on_target",
        "shots_on_target_against": "opponent_shots_on_target",
    }

    processed_groups: List[pd.DataFrame] = []

    for _, group in df.groupby("team", sort=False):
        group = group.sort_values("match_date").copy()

        win_streak_values: List[int] = []
        unbeaten_values: List[int] = []
        win_count = 0
        unbeaten_count = 0
        for res in group["result_class"]:
            win_streak_values.append(win_count)
            unbeaten_values.append(unbeaten_count)
            if pd.isna(res):
                continue
            if res == "win":
                win_count += 1
            else:
                win_count = 0
            if res in {"win", "draw"}:
                unbeaten_count += 1
            else:
                unbeaten_count = 0

        group["feat_win_streak"] = win_streak_values
        group["feat_unbeaten_streak"] = unbeaten_values

        group["feat_form_points_short"] = _ewm(group["team_points_raw"], HALFLIFE_SHORT).fillna(0.0)
        group["feat_form_points_medium"] = _ewm(group["team_points_raw"], HALFLIFE_LONG).fillna(0.0)

        team_xg_series = pd.to_numeric(group.get("team_xg", 0.0), errors="coerce").fillna(0.0)
        opponent_xg_series = pd.to_numeric(group.get("opponent_xg", 0.0), errors="coerce").fillna(0.0)
        attack_ewm = _ewm(team_xg_series, HALFLIFE_SHORT).fillna(team_xg_series.expanding().mean())
        defence_ewm = _ewm(opponent_xg_series, HALFLIFE_SHORT).fillna(opponent_xg_series.expanding().mean())
        group["feat_attack_rating"] = attack_ewm.clip(0.0, 4.0)
        group["feat_defence_rating"] = (1.5 - defence_ewm).clip(-2.0, 2.0)
        team_goals_series = pd.to_numeric(group.get("team_goals", 0.0), errors="coerce").fillna(0.0)
        conversion = team_goals_series.div(team_xg_series.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        overperformance = team_goals_series - team_xg_series
        group["feat_xg_conversion_raw"] = conversion.fillna(1.0).clip(0.0, 4.0)
        group["feat_xg_overperformance_raw"] = overperformance.fillna(0.0).clip(-5.0, 5.0)
        group["feat_xg_conversion_short"] = _ewm(group["feat_xg_conversion_raw"], HALFLIFE_SHORT).fillna(1.0)
        group["feat_xg_conversion_medium"] = _ewm(group["feat_xg_conversion_raw"], HALFLIFE_LONG).fillna(1.0)
        group["feat_xg_overperformance_short"] = _ewm(group["feat_xg_overperformance_raw"], HALFLIFE_SHORT).fillna(0.0)
        group["feat_xg_overperformance_medium"] = _ewm(group["feat_xg_overperformance_raw"], HALFLIFE_LONG).fillna(0.0)

        for metric_name, source_col in metric_sources.items():
            if source_col not in group.columns:
                continue
            series = pd.to_numeric(group[source_col], errors="coerce").fillna(0.0)
            group[f"feat_{metric_name}_ewm_short"] = _ewm(series, HALFLIFE_SHORT).fillna(0.0)
            group[f"feat_{metric_name}_ewm_medium"] = _ewm(series, HALFLIFE_LONG).fillna(0.0)

            home_series = series.where(group["venue"] == "home")
            away_series = series.where(group["venue"] == "away")
            group[f"feat_{metric_name}_home_ewm_short"] = (
                _ewm(home_series, HALFLIFE_SHORT).ffill().fillna(0.0)
            )
            group[f"feat_{metric_name}_away_ewm_short"] = (
                _ewm(away_series, HALFLIFE_SHORT).ffill().fillna(0.0)
            )

        group["feat_xg_diff"] = (
            pd.to_numeric(group.get("team_xg", 0.0), errors="coerce").fillna(0.0)
            - pd.to_numeric(group.get("opponent_xg", 0.0), errors="coerce").fillna(0.0)
        )
        group["feat_goal_diff"] = (
            pd.to_numeric(group.get("team_goals", 0.0), errors="coerce").fillna(0.0)
            - pd.to_numeric(group.get("opponent_goals", 0.0), errors="coerce").fillna(0.0)
        )

        for window in TREND_WINDOWS:
            group[f"feat_xg_trend_{window}"] = _rolling_slope(group["feat_xg_diff"], window).fillna(0.0)
            group[f"feat_points_trend_{window}"] = _rolling_slope(group["team_points_raw"], window).fillna(0.0)
            group[f"feat_goals_trend_{window}"] = _rolling_slope(group["feat_goal_diff"], window).fillna(0.0)

        rest_penalty = (group["feat_rest_days"] - 6.0).fillna(0.0) / 10.0
        group["feat_form_factor"] = (
            group["feat_xg_trend_5"].fillna(0.0)
            + 0.5 * group["feat_points_trend_5"].fillna(0.0)
            - rest_penalty
        ).clip(-2.0, 2.0)

        group["feat_match_style_index"] = (
            group["feat_attack_rating"].fillna(0.0)
            - group["feat_defence_rating"].fillna(0.0)
        ).clip(-4.0, 4.0)

        processed_groups.append(group)

    result = pd.concat(processed_groups, axis=0).sort_index()
    result = _append_dynamic_strength(result)

    feat_columns = [col for col in result.columns if col.startswith("feat_")]
    for col in feat_columns:
        result[col] = result[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return result


def _select_feature_columns(team_history: pd.DataFrame) -> List[str]:
    """List feature columns (prefixed with feat_) to carry into fixture table."""
    return sorted(
        [
            col
            for col in team_history.columns
            if col.startswith("feat_") and col not in EXCLUDED_FEATURE_COLUMNS
        ]
    )


def _build_fixture_rows(team_history: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Convert team history (with new features) into fixture-level dataset."""
    base_cols = [
        "season",
        "match_date",
        "match_id",
        "team",
        "opponent",
        "venue",
        "team_goals",
        "opponent_goals",
        "team_xg",
        "opponent_xg",
        "result_class",
        "odds_win",
        "odds_draw",
        "odds_loss",
    ]

    select_cols = base_cols + feature_columns
    if "rest_days" in team_history.columns:
        select_cols.append("rest_days")
    available_cols = [col for col in select_cols if col in team_history.columns]
    team_history = team_history[available_cols].copy()

    home_rows = team_history[team_history["venue"] == "home"].copy()
    home_rows["rest_days_home"] = home_rows.get("rest_days", np.nan)
    home_rows.rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "team_goals": "home_goals",
            "opponent_goals": "away_goals",
            "team_xg": "home_xg",
            "opponent_xg": "away_xg",
            "result_class": "outcome",
            "odds_win": "home_odds_win",
            "odds_draw": "home_odds_draw",
            "odds_loss": "home_odds_loss",
            "elo_edge": "elo_edge_home",
            "team_elo_pre": "elo_home",
        },
        inplace=True,
    )
    if "rest_days" in home_rows.columns:
        home_rows.drop(columns=["rest_days"], inplace=True)

    away_feature_cols = ["match_id"] + feature_columns + ["rest_days"]
    away_available = [col for col in away_feature_cols if col in team_history.columns]
    away_rows = team_history[team_history["venue"] == "away"][away_available].copy()
    away_rows["rest_days_away"] = away_rows.get("rest_days", np.nan)
    rename_map = {col: f"away_{col}" for col in feature_columns}
    away_rows.rename(columns=rename_map, inplace=True)
    if "rest_days" in away_rows.columns:
        away_rows.drop(columns=["rest_days"], inplace=True)

    fixtures = home_rows.merge(away_rows, on="match_id", how="left")

    for col in feature_columns:
        if col in fixtures.columns:
            fixtures.rename(columns={col: f"home_{col}"}, inplace=True)

    for col in feature_columns:
        base_name = col.replace("feat_", "")
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in fixtures.columns and away_col in fixtures.columns:
            fixtures[f"diff_feat_{base_name}"] = fixtures[home_col] - fixtures[away_col]

    # Rest/context columns
    fixtures["rest_days_home"] = fixtures.get("rest_days_home", np.nan).fillna(5.0)
    fixtures["rest_days_away"] = fixtures.get("rest_days_away", np.nan).fillna(5.0)
    fixtures["rest_days_diff"] = fixtures["rest_days_home"] - fixtures["rest_days_away"]

    chosen_columns = [
        "season",
        "match_date",
        "match_id",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "outcome",
        "home_odds_win",
        "home_odds_draw",
        "home_odds_loss",
        "rest_days_home",
        "rest_days_away",
        "rest_days_diff",
    ]

    home_feature_cols = sorted([f"home_{col}" for col in feature_columns if f"home_{col}" in fixtures.columns])
    away_feature_cols = sorted([f"away_{col}" for col in feature_columns if f"away_{col}" in fixtures.columns])
    diff_feature_cols = sorted([col for col in fixtures.columns if col.startswith("diff_feat_")])

    fixtures = fixtures[
        chosen_columns
        + home_feature_cols
        + away_feature_cols
        + diff_feature_cols
    ].copy()

    fixtures.sort_values(["season", "match_date", "home_team"], inplace=True)
    fixtures.reset_index(drop=True, inplace=True)
    fixtures["is_played"] = fixtures["home_goals"].notna() & fixtures["away_goals"].notna()
    fixtures["label"] = fixtures["outcome"]
    fixtures["fixture_id"] = (
        fixtures["season"].astype(str)
        + "|"
        + fixtures["match_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + fixtures["home_team"]
        + "|"
        + fixtures["away_team"]
    )
    fixtures = fixtures.drop_duplicates(subset=["fixture_id"]).reset_index(drop=True)

    return fixtures


def build_fixture_dataset(team_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct fixture-level dataset and return feature column names.

    Parameters
    ----------
    team_df : pd.DataFrame
        Output of unified_dataset containing one row per team per match.

    Returns
    -------
    fixtures : pd.DataFrame
        Fixture-level dataset (one row per match) with engineered features.
    feature_columns : List[str]
        Names of numeric feature columns (prefixed with home_/away_ where relevant).
    """
    team_history = _prepare_team_history(team_df)
    feature_cols = _select_feature_columns(team_history)
    fixtures = _build_fixture_rows(team_history, feature_cols)

    fixture_feature_cols: List[str] = []
    for base_col in feature_cols:
        h_col = f"home_{base_col}"
        a_col = f"away_{base_col}"
        if h_col in fixtures.columns:
            fixture_feature_cols.append(h_col)
        if a_col in fixtures.columns:
            fixture_feature_cols.append(a_col)
    diff_cols = [col for col in fixtures.columns if col.startswith("diff_feat_")]
    fixture_feature_cols += diff_cols
    fixture_feature_cols += [
        "rest_days_home",
        "rest_days_away",
        "rest_days_diff",
    ]

    return fixtures, fixture_feature_cols


@dataclass
class FixtureDataset:
    fixtures: pd.DataFrame
    feature_columns: List[str]


@dataclass
class UnifiedBuildResult:
    fixtures: pd.DataFrame
    feature_columns: List[str]
    dataset_path: Path
    quality_report_path: Path
    issues_path: Optional[Path]


def _season_label_map(seasons: Iterable) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for season in seasons:
        label = getattr(season, "label", None) or season["label"]
        parts = label.replace("\\", "/").replace("-", "/").split("/")
        try:
            start_year = int(parts[0])
        except (ValueError, IndexError):
            continue
        mapping[str(start_year)] = label
    return mapping


def _ensure_processed_dataset(paths: WorkflowPaths, seasons: List) -> Path:
    processed_path = paths.processed_dir / "epl_rolling_features.csv"
    prepare_dataset(paths, seasons)
    return processed_path


def _season_label_from_date(match_date: pd.Timestamp, season_map: Dict[str, str]) -> Optional[str]:
    if pd.isna(match_date):
        return None
    year = int(match_date.year)
    month = int(match_date.month)
    start_year = year if month >= 7 else year - 1
    return season_map.get(str(start_year))


def _load_processed_football_dataset(processed_path: Path, season_labels: Iterable[str]) -> pd.DataFrame:
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed football-data dataset not found: {processed_path}")
    df = pd.read_csv(processed_path, parse_dates=["match_date"])
    df = df[df["season"].isin(season_labels)].copy()
    if df.empty:
        raise ValueError("No football-data records available for the configured seasons.")

    df["team"] = df["team"].map(canonical_team_name)
    df["opponent"] = df["opponent"].map(canonical_team_name)
    if df["team"].isna().any() or df["opponent"].isna().any():
        raise ValueError("Some football-data team names could not be normalised.")

    df["match_date"] = pd.to_datetime(df["match_date"]).dt.normalize()
    rename_map = {
        "goals_for": "team_goals",
        "goals_against": "opponent_goals",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def _load_upcoming_fixtures(
    paths: WorkflowPaths,
    season_map: Dict[str, str],
    existing_columns: Iterable[str],
) -> pd.DataFrame:
    existing_columns = list(existing_columns)
    fixture_path = paths.raw_dir / UPCOMING_FIXTURES_FILENAME
    if not fixture_path.exists():
        return pd.DataFrame(columns=existing_columns)

    df = pd.read_csv(fixture_path)
    rename_map = {
        "Date": "match_date",
        "MatchDate": "match_date",
        "HomeTeam": "home_team",
        "Home": "home_team",
        "AwayTeam": "away_team",
        "Away": "away_team",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"match_date", "home_team", "away_team"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(
            f"Upcoming fixtures file {fixture_path} missing required columns: {', '.join(sorted(missing))}"
        )

    df["match_date"] = pd.to_datetime(df["match_date"]).dt.normalize()
    fixture_rows: List[dict] = []

    for _, row in df.iterrows():
        season_label = _season_label_from_date(row["match_date"], season_map)
        if season_label is None:
            logger.warning(
                "Skipping upcoming fixture %s vs %s on %s (season not configured).",
                row["home_team"],
                row["away_team"],
                row["match_date"].date(),
            )
            continue

        home = canonical_team_name(row["home_team"])
        away = canonical_team_name(row["away_team"])
        match_id = (
            f"{season_label}_{row['match_date'].strftime('%Y-%m-%d')}_{home}_{away}"
        )

        base_home = {
            "season": season_label,
            "match_date": row["match_date"],
            "team": home,
            "opponent": away,
            "venue": "home",
            "team_goals": np.nan,
            "opponent_goals": np.nan,
            "result": pd.NA,
            "result_class": pd.NA,
            "odds_win": np.nan,
            "odds_draw": np.nan,
            "odds_loss": np.nan,
            "match_id": match_id,
        }
        base_away = {
            "season": season_label,
            "match_date": row["match_date"],
            "team": away,
            "opponent": home,
            "venue": "away",
            "team_goals": np.nan,
            "opponent_goals": np.nan,
            "result": pd.NA,
            "result_class": pd.NA,
            "odds_win": np.nan,
            "odds_draw": np.nan,
            "odds_loss": np.nan,
            "match_id": match_id,
        }
        fixture_rows.append(base_home)
        fixture_rows.append(base_away)

    if not fixture_rows:
        return pd.DataFrame(columns=existing_columns)

    upcoming_df = pd.DataFrame(fixture_rows)
    upcoming_df = upcoming_df.reindex(columns=existing_columns, fill_value=np.nan)
    return upcoming_df


def _load_understat_team_rows(understat_dir: Path, season_map: Dict[str, str], valid_labels: Iterable[str]) -> pd.DataFrame:
    if not understat_dir.exists():
        raise FileNotFoundError(f"Understat directory not found: {understat_dir}")

    rows: List[dict] = []
    valid_set = set(valid_labels)
    for path in sorted(understat_dir.glob("matches_EPL_*.json")):
        season_code = path.stem.split("_")[-1]
        season_label = season_map.get(season_code)
        if season_label is None or season_label not in valid_set:
            continue
        data = json.loads(path.read_text())
        for match in data:
            match_id = int(match["id"])
            match_date = pd.to_datetime(match["datetime"]).tz_localize(None).normalize()
            home = canonical_team_name(match["h"]["title"])
            away = canonical_team_name(match["a"]["title"])
            home_goals = _safe_int(match["goals"].get("h"))
            away_goals = _safe_int(match["goals"].get("a"))
            home_xg = _safe_float(match["xG"].get("h"))
            away_xg = _safe_float(match["xG"].get("a"))

            rows.append(
                {
                    "season": season_label,
                    "team": home,
                    "opponent": away,
                    "venue": "home",
                    "understat_match_id": match_id,
                    "understat_match_date": match_date,
                    "understat_team_goals": home_goals,
                    "understat_opponent_goals": away_goals,
                    "understat_team_xg": home_xg,
                    "understat_opponent_xg": away_xg,
                }
            )
            rows.append(
                {
                    "season": season_label,
                    "team": away,
                    "opponent": home,
                    "venue": "away",
                    "understat_match_id": match_id,
                    "understat_match_date": match_date,
                    "understat_team_goals": away_goals,
                    "understat_opponent_goals": home_goals,
                    "understat_team_xg": away_xg,
                    "understat_opponent_xg": home_xg,
                }
            )

    if not rows:
        raise RuntimeError("No Understat matches were parsed for the configured seasons.")
    df = pd.DataFrame(rows)
    df["understat_match_date"] = pd.to_datetime(df["understat_match_date"]).dt.normalize()
    return df


def _merge_understat_metrics(team_df: pd.DataFrame, understat_df: pd.DataFrame) -> pd.DataFrame:
    merged = team_df.merge(
        understat_df,
        on=["season", "team", "opponent", "venue"],
        how="left",
        validate="one_to_one",
    )

    if "understat_match_date" in merged.columns:
        date_diff = (
            merged["match_date"] - merged["understat_match_date"]
        ).abs().dt.days.fillna(0)
        mismatch_mask = date_diff > 1
        if mismatch_mask.any():
            problematic = merged.loc[
                mismatch_mask,
                ["season", "team", "opponent", "match_date", "understat_match_date"],
            ]
            logger.warning(
                "Understat dates differ from football-data for %d rows; dropping Understat metrics for these fixtures. Sample:\n%s",
                len(problematic),
                problematic.head().to_string(index=False),
            )
            columns_to_clear = [
                "understat_match_id",
                "understat_team_goals",
                "understat_opponent_goals",
                "understat_team_xg",
                "understat_opponent_xg",
            ]
            for col in columns_to_clear:
                if col in merged.columns:
                    merged.loc[mismatch_mask, col] = np.nan
        merged = merged.drop(columns=["understat_match_date"])

    # Validate goal agreement when both sources have values
    if {"understat_team_goals", "understat_opponent_goals"}.issubset(merged.columns):
        goal_mask = merged["team_goals"].notna() & merged["understat_team_goals"].notna()
        mismatch = merged.loc[
            goal_mask & (
                (merged["team_goals"].astype(float) != merged["understat_team_goals"].astype(float))
                | (merged["opponent_goals"].astype(float) != merged["understat_opponent_goals"].astype(float))
            ),
            ["season", "team", "opponent", "match_date", "team_goals", "understat_team_goals"],
        ]
        if not mismatch.empty:
            raise ValueError("Goal totals differ between football-data and Understat. Sample:\n" f"{mismatch.head()}")

    merged["understat_match_id"] = merged["understat_match_id"].astype("Int64")
    merged["team_xg"] = merged["understat_team_xg"]
    merged["opponent_xg"] = merged["understat_opponent_xg"]
    merged = merged.drop(
        columns=[
            "understat_team_goals",
            "understat_opponent_goals",
            "understat_team_xg",
            "understat_opponent_xg",
        ],
        errors="ignore",
    )

    return merged


def _attach_fixture_id(team_df: pd.DataFrame) -> pd.DataFrame:
    df = team_df.copy()
    home_team = np.where(df["venue"] == "home", df["team"], df["opponent"])
    away_team = np.where(df["venue"] == "home", df["opponent"], df["team"])
    df["fixture_id"] = (
        df["season"].astype(str)
        + "|"
        + df["match_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + home_team
        + "|"
        + away_team
    )
    return df


def _validate_team_fixture_counts(team_df: pd.DataFrame) -> None:
    counts = team_df.groupby("fixture_id").size()
    if not counts.eq(2).all():
        bad = counts[counts.ne(2)]
        raise ValueError(f"Each fixture must have exactly two team rows. Found discrepancies for {len(bad)} fixtures.")
    home_dupes = team_df[team_df["venue"] == "home"]["fixture_id"].duplicated()
    if home_dupes.any():
        raise ValueError("Duplicate home entries detected when constructing fixtures.")


def _compute_data_summary(fixtures: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fixtures = fixtures.copy()
    fixtures["match_date"] = pd.to_datetime(fixtures["match_date"]).dt.normalize()

    played_mask = fixtures["home_goals"].notna() & fixtures["away_goals"].notna()
    played = fixtures[played_mask]
    missing_xg_mask = played[
        played["home_xg"].isna() | played["away_xg"].isna()
    ].copy()
    missing_odds_mask = played[
        played[["home_odds_win", "home_odds_draw", "home_odds_loss"]].isna().any(axis=1)
    ].copy()

    summary_rows: List[dict] = []
    for season, group in fixtures.groupby("season"):
        summary_rows.append(
            {
                "season": season,
                "fixture_count": int(len(group)),
                "played_fixture_count": int(group["home_goals"].notna().sum()),
                "upcoming_fixture_count": int(group["home_goals"].isna().sum()),
                "missing_xg_played": int(
                    group[group["home_goals"].notna() & group["home_xg"].isna()].shape[0]
                ),
                "missing_odds_played": int(
                    group[
                        group["home_goals"].notna()
                        & group[["home_odds_win", "home_odds_draw", "home_odds_loss"]].isna().any(axis=1)
                    ].shape[0]
                ),
                "duplicates": int(group["fixture_id"].duplicated().sum()),
                "earliest_match": group["match_date"].min(),
                "latest_match": group["match_date"].max(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["missing_xg_percent"] = summary_df.apply(
            lambda row: (row["missing_xg_played"] / row["played_fixture_count"] * 100.0)
            if row["played_fixture_count"] > 0
            else 0.0,
            axis=1,
        )
        summary_df["missing_odds_percent"] = summary_df.apply(
            lambda row: (row["missing_odds_played"] / row["played_fixture_count"] * 100.0)
            if row["played_fixture_count"] > 0
            else 0.0,
            axis=1,
        )
        summary_df["earliest_match"] = summary_df["earliest_match"].dt.strftime("%Y-%m-%d")
        summary_df["latest_match"] = summary_df["latest_match"].dt.strftime("%Y-%m-%d")

    issues_frames = []
    if not missing_xg_mask.empty:
        missing_xg_mask = missing_xg_mask.assign(issue="missing_xg")
        issues_frames.append(
            missing_xg_mask[
                ["season", "match_date", "home_team", "away_team", "home_goals", "away_goals", "issue"]
            ]
        )
    if not missing_odds_mask.empty:
        missing_odds_mask = missing_odds_mask.assign(issue="missing_odds")
        issues_frames.append(
            missing_odds_mask[
                ["season", "match_date", "home_team", "away_team", "home_odds_win", "home_odds_draw", "home_odds_loss", "issue"]
            ]
        )

    issues_df = pd.concat(issues_frames, ignore_index=True) if issues_frames else pd.DataFrame()
    return summary_df, issues_df


def _write_feature_columns_metadata(feature_columns: List[str], processed_dir: Path) -> Path:
    metadata_path = processed_dir / "epl_unified_feature_columns.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(feature_columns, indent=2))
    return metadata_path


def build_unified_dataset() -> UnifiedBuildResult:
    config = default_config()
    project_root = Path(__file__).resolve().parent
    paths = WorkflowPaths.default(project_root)
    seasons = seasons_from_config(config)
    season_labels = [season.label for season in seasons]

    processed_path = _ensure_processed_dataset(paths, seasons)
    team_df = _load_processed_football_dataset(processed_path, season_labels)

    season_map = _season_label_map(seasons)
    upcoming_df = _load_upcoming_fixtures(paths, season_map, team_df.columns)
    if not upcoming_df.empty:
        existing_keys = set(
            zip(team_df["season"], team_df["team"], team_df["opponent"], team_df["venue"])
        )
        upcoming_keys = list(
            zip(upcoming_df["season"], upcoming_df["team"], upcoming_df["opponent"], upcoming_df["venue"])
        )
        mask = [
            key not in existing_keys
            for key in upcoming_keys
        ]
        upcoming_df = upcoming_df.loc[mask]
        if upcoming_df.empty:
            logger.info(
                "Upcoming fixtures file processed but all rows match existing fixtures; nothing to append."
            )
        else:
            before = len(team_df)
            team_df = pd.concat([team_df, upcoming_df], ignore_index=True)
            team_df = team_df.drop_duplicates(
                subset=["season", "match_date", "team", "opponent", "venue"],
                keep="first",
            )
            added = len(team_df) - before
            logger.info(
                "Added %d upcoming team rows from %s.",
                added,
                UPCOMING_FIXTURES_FILENAME,
            )

    understat_dir = project_root / "football_weight_model" / "data" / "raw" / "understat"
    understat_df = _load_understat_team_rows(understat_dir, season_map, season_labels)

    merged = _merge_understat_metrics(team_df, understat_df)
    merged = _attach_fixture_id(merged)
    _validate_team_fixture_counts(merged)

    merged["team_xg"] = merged["team_xg"].astype(float)
    merged["opponent_xg"] = merged["opponent_xg"].astype(float)

    fixtures, feature_columns = build_fixture_dataset(merged)
    fixtures["fixture_id"] = (
        fixtures["season"].astype(str)
        + "|"
        + fixtures["match_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + fixtures["home_team"]
        + "|"
        + fixtures["away_team"]
    )

    if fixtures["fixture_id"].duplicated().any():
        dupes = fixtures[fixtures["fixture_id"].duplicated()]["fixture_id"].unique()
        raise ValueError(f"Duplicate fixture identifiers detected: {dupes[:5]}")

    summary_df, issues_df = _compute_data_summary(fixtures)
    duplicates_total = int(summary_df["duplicates"].sum()) if not summary_df.empty else 0

    outputs_dir = paths.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    quality_path = outputs_dir / "data_quality_report.csv"
    summary_df.to_csv(quality_path, index=False)

    issues_path: Optional[Path] = outputs_dir / "data_issues.csv"
    if issues_df.empty:
        if issues_path.exists():
            issues_path.unlink()
        issues_path = None
    else:
        issues_df.to_csv(issues_path, index=False)

    played_issues = summary_df[
        (summary_df["missing_xg_played"] > 0) | (summary_df["missing_odds_played"] > 0)
    ]
    if duplicates_total > 0 or not played_issues.empty:
        raise ValueError(
            "Data quality checks failed. See outputs/data_quality_report.csv"
            f"{' and outputs/data_issues.csv' if issues_path else ''} for details."
        )

    dataset_path = paths.processed_dir / "epl_unified_team_matches.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(dataset_path, index=False)

    _write_feature_columns_metadata(feature_columns, paths.processed_dir)

    return UnifiedBuildResult(
        fixtures=fixtures,
        feature_columns=feature_columns,
        dataset_path=dataset_path,
        quality_report_path=quality_path,
        issues_path=issues_path,
    )
