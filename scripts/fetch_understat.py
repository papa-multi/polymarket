#!/usr/bin/env python3
"""Fetch Premier League match-level data from Understat using curl + HTML parsing."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

LEAGUE = "EPL"
SEASONS: Iterable[int] = range(2021, 2026)
PAGE_URL_TEMPLATE = "https://understat.com/league/{league}/{season}"
CURL_HEADERS = [
    "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language: en-US,en;q=0.9",
    "Referer: https://understat.com/",
]
OUTPUT_DIR = Path("football_weight_model/data/raw/understat")


def fetch_html(season: int) -> str:
    url = PAGE_URL_TEMPLATE.format(league=LEAGUE, season=season)
    cmd = ["curl", "-sS"]
    for header in CURL_HEADERS:
        cmd.extend(["-H", header])
    cmd.append(url)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def extract_matches_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")
    pattern = re.compile(
        r"var\s+(?P<var_name>[A-Za-z0-9_]+)\s*=\s*JSON\.parse\((?P<quote>['\"])(?P<data>.*?)(?P=quote)\)",
        re.DOTALL,
    )
    decoded_dates = None
    for script in scripts:
        for match in pattern.finditer(script.text):
            var_name = match.group("var_name")
            json_text = match.group("data")
            decoded = bytes(json_text, "utf-8").decode("unicode_escape")
            payload = json.loads(decoded)
            if var_name == "matchesData":
                return payload
            if var_name == "datesData" and decoded_dates is None:
                # Fall back to datesData when matchesData has been renamed
                decoded_dates = payload
    if decoded_dates is not None:
        return decoded_dates
    raise ValueError("matches/dates data not found in page")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        print(f"Fetching {LEAGUE} {season}...")
        html = fetch_html(season)
        matches = extract_matches_from_html(html)
        out_file = OUTPUT_DIR / f"matches_{LEAGUE}_{season}.json"
        out_file.write_text(json.dumps(matches), encoding="utf-8")
        print(f"Saved {out_file} ({len(matches)} matches)")


if __name__ == "__main__":
    main()
