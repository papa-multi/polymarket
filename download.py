"""
Utilities for pulling raw Premier League data from football-data.co.uk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import logging

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"


@dataclass(frozen=True)
class SeasonSpec:
    label: str  # e.g. "2021-2022"
    code: str   # e.g. "2122" used in URL paths


def download_season(season: SeasonSpec, dest_folder: Path) -> Path:
    dest_folder.mkdir(parents=True, exist_ok=True)
    target = dest_folder / f"epl_{season.code}.csv"
    url = BASE_URL.format(season=season.code)

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def download_multiple(seasons: Iterable[SeasonSpec], dest_folder: Path) -> None:
    for season in seasons:
        try:
            download_season(season, dest_folder)
        except Exception as exc:  # pragma: no cover - logging side effect
            logging.error("Failed to download %s: %s", season.label, exc)
            raise

