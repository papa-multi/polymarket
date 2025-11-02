"""Sequential online outcome model built on incremental logistic regression."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

ResultClass = str  # expected values: "win", "draw", "loss"


@dataclass
class MatchPrediction:
    season: str
    team: str
    opponent: str
    match_index: int
    match_id: str
    match_date: pd.Timestamp
    predicted: Optional[ResultClass]
    actual: Optional[ResultClass]
    correct: Optional[bool]
    model_scores: Dict[ResultClass, float]
    model_probabilities: Dict[ResultClass, float]
    bookmaker_probabilities: Optional[Dict[ResultClass, float]] = None
    expected_goal_probabilities: Optional[Dict[ResultClass, float]] = None


@dataclass
class OnlineOutcomeModel:
    feature_names: Sequence[str]
    learning_rate: float = 0.03
    bookmaker_blend: float = 0.18
    bookmaker_decay: float = 0.12
    expected_goal_blend: float = 0.45
    recency_rate: float = 0.015
    draw_squeeze: float = 0.9
    warmup_matches: int = 5
    min_global_samples: int = 0
    classes: Sequence[ResultClass] = field(default=("win", "draw", "loss"))
    baseline_prob_columns: Mapping[ResultClass, str] = field(default_factory=dict)
    random_state: int = 21

    def __post_init__(self) -> None:
        self.label_encoder = LabelEncoder().fit(list(self.classes))
        self._class_indices = np.arange(len(self.classes))
        self.scaler = StandardScaler(with_mean=False)
        self._scaler_initialized = False
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="invscaling",
            eta0=self.learning_rate,
            power_t=0.5,
            random_state=self.random_state,
        )
        self._model_initialized = False
        self.team_match_counts: Dict[tuple[str, str], int] = {}
        self.samples_seen: int = 0
        self.class_counts: Dict[ResultClass, int] = {cls: 0 for cls in self.classes}
        if self.min_global_samples <= 0:
            self.min_global_samples = max(50, self.warmup_matches * 20)

    def _baseline_from_row(self, row: pd.Series) -> Optional[np.ndarray]:
        if not self.baseline_prob_columns:
            return None
        baseline = []
        for cls in self.classes:
            col = self.baseline_prob_columns.get(cls)
            if not col or col not in row.index:
                return None
            val = row[col]
            if pd.isna(val):
                return None
            baseline.append(float(val))
        baseline = np.array(baseline, dtype=float)
        total = baseline.sum()
        if total <= 0:
            return None
        return baseline / total

    def _blend_with_bookmaker(
        self,
        model_probs: np.ndarray,
        book_probs: Optional[Dict[ResultClass, float]],
        *,
        weight: Optional[float] = None,
    ) -> np.ndarray:
        if not book_probs:
            return model_probs
        blend = np.array([book_probs.get(cls, 0.0) for cls in self.classes], dtype=float)
        total = blend.sum()
        if total <= 0:
            return model_probs
        blend /= total
        mix = self.bookmaker_blend if weight is None else max(0.0, min(weight, 0.6))
        if mix <= 0:
            return model_probs
        combined = (1.0 - mix) * model_probs + mix * blend
        combined_sum = combined.sum()
        if combined_sum <= 0:
            return model_probs
        return combined / combined_sum

    def _dynamic_book_weight(self, team_count: int, required_warmup: int) -> float:
        if self.bookmaker_blend <= 0:
            return 0.0
        if team_count <= required_warmup:
            return self.bookmaker_blend * 0.5
        decay_steps = max(0, team_count - required_warmup)
        return self.bookmaker_blend * float(np.exp(-self.bookmaker_decay * decay_steps))

    def run_season(
        self,
        season: str,
        matches: pd.DataFrame,
        warmup_matches: Optional[int] = None,
    ) -> list[MatchPrediction]:
        predictions: list[MatchPrediction] = []
        sorted_matches = matches.sort_values(["match_date", "team"])

        required_warmup = warmup_matches if warmup_matches is not None else self.warmup_matches

        for _, row in sorted_matches.iterrows():
            team = str(row.get("team", ""))
            opponent = str(row.get("opponent", ""))
            match_date = pd.to_datetime(row["match_date"])
            match_id = str(row.get("match_id", ""))
            team_match_number = int(row.get("team_match_number", 0))
            actual_result = row.get("result_class")

            team_key = (season, team)
            team_count = self.team_match_counts.get(team_key, 0)

            feature_vector = row[self.feature_names].to_numpy(dtype=float, copy=True)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            feature_vector_2d = feature_vector.reshape(1, -1)

            self.scaler.partial_fit(feature_vector_2d)
            self._scaler_initialized = True
            features_scaled = self.scaler.transform(feature_vector_2d)

            baseline = self._baseline_from_row(row)
            book_probs = None
            if {"book_prob_win", "book_prob_draw", "book_prob_loss"}.issubset(row.index):
                raw_book_probs = {
                    "win": row["book_prob_win"],
                    "draw": row["book_prob_draw"],
                    "loss": row["book_prob_loss"],
                }
                book_probs = {
                    outcome: float(val) if pd.notna(val) else 0.0
                    for outcome, val in raw_book_probs.items()
                }

            expected_goal_probs = None
            eg_columns = [f"eg_prob_{cls}" for cls in self.classes]
            if all(col in row.index and pd.notna(row[col]) for col in eg_columns):
                expected_goal_probs = {cls: float(row[f"eg_prob_{cls}"]) for cls in self.classes}

            can_predict = (
                self._model_initialized
                and team_count >= required_warmup
                and self.samples_seen >= self.min_global_samples
            )

            probabilities = np.full(len(self.classes), np.nan, dtype=float)
            predicted_class: Optional[ResultClass] = None

            baseline_from_book = None
            if book_probs is not None:
                baseline_from_book = np.array([book_probs.get(cls, 0.0) for cls in self.classes], dtype=float)
                total = baseline_from_book.sum()
                if total > 0:
                    baseline_from_book = baseline_from_book / total
                else:
                    baseline_from_book = None

            logistic_probs = None
            if self._model_initialized:
                logistic_probs = self.model.predict_proba(features_scaled)[0]
                lp_sum = logistic_probs.sum()
                if lp_sum > 0:
                    logistic_probs = logistic_probs / lp_sum

            if logistic_probs is not None and can_predict:
                model_probs = logistic_probs
            elif baseline is not None:
                model_probs = baseline
            elif baseline_from_book is not None:
                model_probs = baseline_from_book
            else:
                model_probs = np.full(len(self.classes), 1.0 / len(self.classes), dtype=float)

            if expected_goal_probs is not None:
                eg_vector = np.array([expected_goal_probs.get(cls, 0.0) for cls in self.classes], dtype=float)
                eg_sum = eg_vector.sum()
                if eg_sum > 0:
                    eg_vector /= eg_sum
                    eg_weight = self.expected_goal_blend if can_predict else self.expected_goal_blend * 0.5
                    if baseline_from_book is None:
                        eg_weight = min(0.75, eg_weight * 1.5)
                    team_eff = row.get("team_xg_conversion_5")
                    opp_eff = row.get("opponent_xg_conversion_5")
                    if pd.notna(team_eff) and pd.notna(opp_eff):
                        opp_eff_val = float(opp_eff)
                        ratio = float(team_eff) / max(opp_eff_val, 1e-3)
                        ratio = float(np.clip(ratio, 0.45, 1.8))
                        efficiency_scale = np.array([ratio, 1.0, 1.0 / ratio], dtype=float)
                        eg_vector = eg_vector * efficiency_scale
                        eg_vector = np.clip(eg_vector, 1e-6, None)
                        eg_vector /= eg_vector.sum()
                        eg_weight = eg_weight * (1.0 + 0.2 * np.tanh(ratio - 1.0))
                    over_diff = row.get("xg_overperformance_diff_5")
                    if pd.notna(over_diff):
                        eg_weight = eg_weight * (1.0 + 0.12 * np.tanh(float(over_diff)))
                    eg_weight = max(0.0, min(eg_weight, 0.75))
                    model_probs = (1.0 - eg_weight) * model_probs + eg_weight * eg_vector

            if book_probs is not None:
                blend_weight = self._dynamic_book_weight(team_count, required_warmup)
                model_probs = self._blend_with_bookmaker(model_probs, book_probs, weight=blend_weight)

            probabilities = np.array(model_probs, dtype=float)
            if np.isfinite(probabilities).all():
                probs_sum = probabilities.sum()
                if probs_sum > 0:
                    probabilities = probabilities / probs_sum
                if "draw" in self.classes and 0 <= self.draw_squeeze < 1.0:
                    draw_idx = self.classes.index("draw")
                    draw_prob = probabilities[draw_idx]
                    if np.isfinite(draw_prob) and draw_prob > 0:
                        squeeze = float(np.clip(self.draw_squeeze, 0.0, 1.0))
                        removed = draw_prob * (1.0 - squeeze)
                        probabilities[draw_idx] = draw_prob * squeeze
                        other_idx = [i for i in range(len(self.classes)) if i != draw_idx]
                        other_sum = probabilities[other_idx].sum()
                        if other_sum > 0 and removed > 0:
                            share = probabilities[other_idx] / other_sum
                            probabilities[other_idx] += share * removed
                        else:
                            probabilities[draw_idx] += removed
                        total = probabilities.sum()
                        if total > 0:
                            probabilities = probabilities / total
                predicted_class = self.classes[int(np.argmax(probabilities))]
            else:
                predicted_class = None

            model_scores = {
                cls: float(np.log(np.clip(probabilities[idx], 1e-12, 1.0))) if np.isfinite(probabilities[idx]) else np.nan
                for idx, cls in enumerate(self.classes)
            }
            model_probabilities = {
                cls: float(probabilities[idx]) if np.isfinite(probabilities[idx]) else np.nan
                for idx, cls in enumerate(self.classes)
            }

            correct = None
            if predicted_class is not None and actual_result in self.classes:
                correct = predicted_class == actual_result

            predictions.append(
                MatchPrediction(
                    season=season,
                    team=team,
                    opponent=opponent,
                    match_index=team_match_number,
                    match_id=match_id,
                    match_date=match_date,
                    predicted=predicted_class,
                    actual=actual_result,
                    correct=correct,
                    model_scores=model_scores,
                    model_probabilities=model_probabilities,
                    bookmaker_probabilities=book_probs,
                    expected_goal_probabilities=expected_goal_probs,
                )
            )

            if actual_result in self.classes:
                label = self.label_encoder.transform([actual_result])[0]
                self.class_counts[actual_result] += 1
                self.samples_seen += 1
                freq = self.class_counts[actual_result] / max(self.samples_seen, 1)
                weight = 1.0 / max(freq, 1e-3)
                if self.recency_rate > 0:
                    recent_factor = float(np.exp(self.recency_rate * max(team_count - required_warmup, 0)))
                    weight *= recent_factor
                if not self._model_initialized:
                    self.model.partial_fit(
                        features_scaled,
                        [label],
                        classes=self._class_indices,
                        sample_weight=[weight],
                    )
                    self._model_initialized = True
                else:
                    self.model.partial_fit(features_scaled, [label], sample_weight=[weight])

            self.team_match_counts[team_key] = team_count + 1

        return predictions
