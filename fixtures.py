"""
Fixture table builder for clean home/away scheduling information.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd

from .team_names import canonical_team_name


@dataclass
class FixturePaths:
    processed_matches: Path
    fixtures_out: Path
    understat_dir: Path

    @classmethod
    def default(cls, root: Path) -> "FixturePaths":
        return cls(
            processed_matches=root / "data" / "processed" / "epl_unified_team_matches.csv",
            fixtures_out=root / "data" / "processed" / "epl_fixtures.csv",
            understat_dir=root / "football_weight_model" / "data" / "raw" / "understat",
        )


def _implied_probabilities(row: pd.Series) -> tuple[Optional[float], Optional[float], Optional[float]]:
    odds_cols = ["odds_win", "odds_draw", "odds_loss"]
    if not all(col in row.index for col in odds_cols):
        return (None, None, None)
    odds = row[odds_cols].to_numpy(dtype=float)
    if np.any(odds <= 0) or np.any(np.isnan(odds)):
        return (None, None, None)
    inv = 1.0 / odds
    overround = inv.sum()
    if overround <= 0:
        return (None, None, None)
    probs = inv / overround
    return tuple(float(p) for p in probs)


def _load_future_understat(understat_dir: Path) -> pd.DataFrame:
    if not understat_dir.exists():
        return pd.DataFrame(
            columns=["season", "match_date", "home_team", "away_team", "understat_match_id"]
        )

    records = []
    for path in sorted(understat_dir.glob("matches_EPL_*.json")):
        season_code = path.stem.split("_")[-1]
        try:
            season_start = int(season_code)
        except ValueError:
            continue
        season_label = f"{season_start}-{season_start + 1}"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for match in data:
            goals = match.get("goals", {})
            if goals.get("h") is not None or goals.get("a") is not None:
                continue  # already played
            match_date = pd.to_datetime(match["datetime"]).normalize()
            home = canonical_team_name(match["h"]["title"])
            away = canonical_team_name(match["a"]["title"])
            records.append(
                {
                    "season": season_label,
                    "match_date": match_date,
                    "home_team": home,
                    "away_team": away,
                    "understat_match_id": int(match["id"]),
                }
            )
    if not records:
        return pd.DataFrame(
            columns=["season", "match_date", "home_team", "away_team", "understat_match_id"]
        )
    return pd.DataFrame.from_records(records)


def build_fixture_table(paths: FixturePaths | None = None) -> Path:
    paths = paths or FixturePaths.default(Path(__file__).resolve().parent)
    df = pd.read_csv(paths.processed_matches, parse_dates=["match_date"])
    df["team"] = df["team"].map(canonical_team_name)
    df["opponent"] = df["opponent"].map(canonical_team_name)

    home_rows = df[df["venue"].str.lower() == "home"].copy()
    if home_rows.empty:
        raise ValueError("No home rows found in the processed dataset.")

    fixture_cols = [
        "match_id",
        "season",
        "match_date",
        "team",
        "opponent",
        "result_class",
        "team_goals",
        "opponent_goals",
        "team_xg",
        "opponent_xg",
        "rest_days",
    ]
    available_cols = [col for col in fixture_cols if col in home_rows.columns]
    fixtures = home_rows[available_cols].copy()
    fixtures.rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "team_goals": "home_goals",
            "opponent_goals": "away_goals",
            "team_xg": "home_xg",
            "opponent_xg": "away_xg",
        },
        inplace=True,
    )

    # Ensure rest_days exists (fallback to NaN if missing)
    if "rest_days" not in fixtures.columns:
        fixtures["rest_days"] = np.nan
    fixtures.rename(columns={"rest_days": "home_rest_days"}, inplace=True)

    # Add away rest days from away rows if available
    if "rest_days" in df.columns:
        away_rest = (
            df[df["venue"].str.lower() == "away"][["match_id", "rest_days"]]
            .rename(columns={"rest_days": "away_rest_days"})
        )
        fixtures = fixtures.merge(away_rest, on="match_id", how="left")
    else:
        fixtures["away_rest_days"] = np.nan

    # Attach bookmaker implied probabilities
    probs = home_rows.apply(_implied_probabilities, axis=1, result_type="expand")
    fixtures[["prob_home_win_market", "prob_draw_market", "prob_away_win_market"]] = probs
    book_cols = ["book_prob_win", "book_prob_draw", "book_prob_loss"]
    if set(book_cols).issubset(home_rows.columns):
        fallback = home_rows[book_cols].to_numpy(dtype=float)
        for idx, col in enumerate(["prob_home_win_market", "prob_draw_market", "prob_away_win_market"]):
            fallback_series = pd.Series(fallback[:, idx], index=fixtures.index, dtype=float)
            fixtures[col] = fixtures[col].fillna(fallback_series)
    fixtures[["prob_home_win_market", "prob_draw_market", "prob_away_win_market"]] = fixtures[
        ["prob_home_win_market", "prob_draw_market", "prob_away_win_market"]
    ].fillna(1.0 / 3.0)

    # Append future fixtures from Understat (goals not yet recorded)
    future_understat = _load_future_understat(paths.understat_dir)
    if not future_understat.empty:
        key_cols = ["season", "match_date", "home_team", "away_team"]
        existing = fixtures[key_cols]
        future_understat = (
            future_understat.merge(existing, on=key_cols, how="left", indicator=True)
            .loc[lambda df: df["_merge"] == "left_only", key_cols + ["understat_match_id"]]
        )
        if not future_understat.empty:
            future_understat = future_understat.drop_duplicates(subset=key_cols)
            future_understat["home_rest_days"] = np.nan
            future_understat["away_rest_days"] = np.nan
            future_understat["prob_home_win_market"] = np.nan
            future_understat["prob_draw_market"] = np.nan
            future_understat["prob_away_win_market"] = np.nan
            if "match_id" in fixtures.columns:
                future_understat["match_id"] = (
                    future_understat["season"]
                    + "_"
                    + future_understat["match_date"].dt.strftime("%Y-%m-%d")
                    + "_"
                    + future_understat["home_team"]
                    + "_"
                    + future_understat["away_team"]
                )
            for column in fixtures.columns:
                if column not in future_understat.columns:
                    future_understat[column] = np.nan
            future_understat = future_understat[fixtures.columns]
            fixtures = pd.concat([fixtures, future_understat], ignore_index=True, sort=False)

    # Sort and save
    fixtures = fixtures.sort_values(["season", "match_date", "home_team"]).reset_index(drop=True)
    fixtures.to_csv(paths.fixtures_out, index=False)
    return paths.fixtures_out
