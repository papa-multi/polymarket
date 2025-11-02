"""
Load the football pipeline configuration from YAML so command-line tools can
stay declarative.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class FootballConfig:
    raw: Dict[str, Any]

    @property
    def seasons(self) -> List[Dict[str, str]]:
        return self.raw["data"]["seasons"]

    @property
    def learning_rate(self) -> float:
        return float(self.raw["model"]["learning_rate"])

    @property
    def warmup_matches(self) -> int:
        return int(self.raw["data"]["warmup_matches"])

    @property
    def feature_weights(self) -> Dict[str, str]:
        base = self.raw["features"]["base_columns"]
        engineered = self.raw["features"]["engineered"]
        weights = {name: spec["weight"] for name, spec in base.items()}
        weights.update({name: spec["weight"] for name, spec in engineered.items()})
        return weights


def load_config(path: Path | None = None) -> FootballConfig:
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return FootballConfig(raw=raw)

