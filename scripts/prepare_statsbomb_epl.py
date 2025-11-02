#!/usr/bin/env python3
"""Prepare Premier League match data (including xG) from the StatsBomb Open Data dump."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class StatsBombSeason:
    season_name: str
    season_id: int


PREMIER_LEAGUE_COMPETITION_ID = 2
SEASONS: List[StatsBombSeason] = [
    StatsBombSeason("2003/2004", 44),
    StatsBombSeason("2015/2016", 27),
]


def load_matches(data_root: Path, season: StatsBombSeason) -> List[Dict]:
    matches_path = data_root / "matches" / str(PREMIER_LEAGUE_COMPETITION_ID) / f"{season.season_id}.json"
    if not matches_path.exists():
        raise FileNotFoundError(f"Matches file not found: {matches_path}")
    with matches_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_events(data_root: Path, match_id: int) -> List[Dict]:
    events_path = data_root / "events" / f"{match_id}.json"
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")
    with events_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarise_match(match: Dict, events: List[Dict], season_name: str) -> Dict:
    match_id = match["match_id"]
    home_team = match["home_team"]["home_team_name"]
    away_team = match["away_team"]["away_team_name"]
    home_team_id = match["home_team"]["home_team_id"]
    away_team_id = match["away_team"]["away_team_id"]

    home_goals = match["home_score"]
    away_goals = match["away_score"]

    home_xg = away_xg = 0.0
    for event in events:
        if event.get("type", {}).get("name") != "Shot":
            continue
        team = event["team"]["id"]
        xg_value = event.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
        if team == home_team_id:
            home_xg += xg_value
        elif team == away_team_id:
            away_xg += xg_value

    return {
        "match_id": match_id,
        "season": season_name,
        "match_date": match["match_date"],
        "kick_off": match.get("kick_off"),
        "stadium": match.get("stadium", {}).get("name"),
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "competition_stage": match.get("competition_stage", {}).get("name"),
        "referee": match.get("referee", {}).get("name"),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw" / "statsbomb" / "open-data-master" / "data"
    if not data_root.exists():
        raise FileNotFoundError(
            f"StatsBomb data directory not found: {data_root}\n"
            "Make sure you extracted the open-data archive under data/raw/statsbomb/."
        )

    match_records = []
    for season in SEASONS:
        matches = load_matches(data_root, season)
        for match in matches:
            events = load_events(data_root, match["match_id"])
            match_records.append(summarise_match(match, events, season.season_name))

    matches_df = (
        pd.DataFrame(match_records)
        .sort_values(["season", "match_date", "match_id"])
        .reset_index(drop=True)
    )

    team_rows = []
    for record in match_records:
        match_id = record["match_id"]
        season_name = record["season"]
        date = record["match_date"]
        for side in ("home", "away"):
            team_rows.append(
                {
                    "match_id": match_id,
                    "season": season_name,
                    "match_date": date,
                    "team": record[f"{side}_team"],
                    "opponent": record["away_team"] if side == "home" else record["home_team"],
                    "team_goals": record[f"{side}_goals"],
                    "opponent_goals": record["away_goals"] if side == "home" else record["home_goals"],
                    "team_xg": record[f"{side}_xg"],
                    "opponent_xg": record["away_xg"] if side == "home" else record["home_xg"],
                    "venue": "home" if side == "home" else "away",
                }
            )
    team_df = (
        pd.DataFrame(team_rows)
        .sort_values(["season", "match_date", "team", "match_id"])
        .reset_index(drop=True)
    )

    output_dir = project_root / "data" / "interim" / "statsbomb"
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_df.to_csv(output_dir / "epl_matches.csv", index=False)
    team_df.to_csv(output_dir / "epl_team_matches.csv", index=False)

    print(f"Created {len(matches_df)} match rows and {len(team_df)} team rows.")
    print(f"Match-level data: {output_dir / 'epl_matches.csv'}")
    print(f"Team-level data : {output_dir / 'epl_team_matches.csv'}")


if __name__ == "__main__":
    main()
