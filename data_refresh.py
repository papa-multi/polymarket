"""
Refresh football-data and Understat sources (download only; no dataset build).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Iterable, List

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

from .workflow import WorkflowPaths, default_config, seasons_from_config
from .download import download_multiple

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = "EPL"
UNDERSTAT_SEASONS: Iterable[int] = range(2021, 2026)


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def clean_existing_data(paths: WorkflowPaths) -> None:
    """Remove processed/interim/outputs while preserving Understat downloads."""
    for target in [
        paths.base_dir / "data" / "interim",
        paths.base_dir / "data" / "processed",
        paths.outputs_dir,
    ]:
        _safe_rmtree(target)

    raw_dir = paths.raw_dir
    if raw_dir.exists():
        for child in raw_dir.iterdir():
            if child.name == "understat":
                continue
            if child.name == "upcoming_fixtures.csv":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def download_understat_season(season: int, output_dir: Path) -> Path:
    if BeautifulSoup is None:
        raise ImportError("beautifulsoup4 is required to download Understat data.")

    url = f"https://understat.com/league/{UNDERSTAT_LEAGUE}/{season}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://understat.com/",
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    scripts = soup.find_all("script")
    pattern = re.compile(
        r"var\s+(?P<var_name>[A-Za-z0-9_]+)\s*=\s*JSON\.parse\((?P<quote>['\"])(?P<data>.*?)(?P=quote)\)",
        re.DOTALL,
    )
    matches_payload = None
    for script in scripts:
        for match in pattern.finditer(script.text):
            var_name = match.group("var_name")
            raw_json = match.group("data")
            decoded = bytes(raw_json, "utf-8").decode("unicode_escape")
            payload = json.loads(decoded)
            if var_name == "matchesData":
                matches_payload = payload
                break
        if matches_payload is not None:
            break

    if matches_payload is None:
        raise ValueError(f"Unable to locate matches data for season {season}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"matches_{UNDERSTAT_LEAGUE}_{season}.json"
    out_path.write_text(json.dumps(matches_payload), encoding="utf-8")
    return out_path


def refresh_understat(output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(output_dir.glob("matches_EPL_*.json"))
    if cached:
        logger.info("Using existing Understat files (skipping download).")
        return cached

    refreshed: List[Path] = []
    errors: List[str] = []
    for season in UNDERSTAT_SEASONS:
        try:
            refreshed.append(download_understat_season(season, output_dir))
        except Exception as exc:  # pragma: no cover
            errors.append(f"{season}: {exc}")

    if errors:
        logger.warning("Understat download issues: %s", "; ".join(errors))
    if not refreshed:
        cached = sorted(output_dir.glob("matches_EPL_*.json"))
        if not cached:
            raise RuntimeError("Understat data unavailable and no cached files found.")
        logger.info("Falling back to existing Understat files under %s", output_dir)
        return cached
    return refreshed


def refresh_all_sources(delete_existing: bool = True) -> None:
    config = default_config()
    project_root = Path(__file__).resolve().parent
    workflow_paths = WorkflowPaths.default(project_root)
    seasons = seasons_from_config(config)

    if delete_existing:
        clean_existing_data(workflow_paths)

    download_multiple(seasons, workflow_paths.raw_dir)
    logger.info("Football-data raw CSVs refreshed under %s", workflow_paths.raw_dir)

    understat_dir = project_root / "football_weight_model" / "data" / "raw" / "understat"
    refresh_understat(understat_dir)
    logger.info("Understat data refreshed under %s", understat_dir)
