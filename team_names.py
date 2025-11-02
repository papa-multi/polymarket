"""Canonical team name mapping utilities."""
from __future__ import annotations

from typing import Dict

_TEAM_ALIASES: Dict[str, str] = {
    "man united": "Manchester United",
    "manchester united": "Manchester United",
    "man utd": "Manchester United",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "spurs": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "west ham": "West Ham United",
    "west ham united": "West Ham United",
    "west ham utd": "West Ham United",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United",
    "brighton": "Brighton & Hove Albion",
    "brighton & hove albion": "Brighton & Hove Albion",
    "brighton and hove albion": "Brighton & Hove Albion",
    "leicester": "Leicester City",
    "leicester city": "Leicester City",
    "norwich": "Norwich City",
    "norwich city": "Norwich City",
    "leeds": "Leeds United",
    "leeds united": "Leeds United",
    "sheffield utd": "Sheffield United",
    "sheffield united": "Sheffield United",
    "cardiff": "Cardiff City",
    "cardiff city": "Cardiff City",
    "birmingham": "Birmingham City",
    "birmingham city": "Birmingham City",
    "stoke": "Stoke City",
    "stoke city": "Stoke City",
    "swansea": "Swansea City",
    "swansea city": "Swansea City",
    "hull": "Hull City",
    "hull city": "Hull City",
    "qpr": "Queens Park Rangers",
    "queens park rangers": "Queens Park Rangers",
    "nottingham forest": "Nottingham Forest",
    "nott'm forest": "Nottingham Forest",
    "ipswich": "Ipswich Town",
    "ipswich town": "Ipswich Town",
    "bournemouth": "AFC Bournemouth",
    "afc bournemouth": "AFC Bournemouth",
    "west brom": "West Bromwich Albion",
    "west bromwich albion": "West Bromwich Albion",
    "brentford": "Brentford",
    "aston villa": "Aston Villa",
    "fulham": "Fulham",
    "everton": "Everton",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "liverpool": "Liverpool",
    "crystal palace": "Crystal Palace",
    "burnley": "Burnley",
    "southampton": "Southampton",
    "watford": "Watford",
    "derby": "Derby County",
    "derby county": "Derby County",
    "notts county": "Notts County",
}


def canonical_team_name(name: str) -> str:
    """Return the canonical team name for the provided alias."""
    if name is None:
        return name
    key = name.strip().lower()
    return _TEAM_ALIASES.get(key, name.strip())

