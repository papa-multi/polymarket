"""Generate diagnostic reports for model outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from .workflow import WorkflowPaths


def _safe_accuracy(preds: pd.Series, actual: pd.Series) -> float:
    mask = preds.notna() & actual.notna()
    if mask.sum() == 0:
        return float("nan")
    return float(accuracy_score(actual.loc[mask], preds.loc[mask]))


def _safe_brier(prob_matrix: np.ndarray, actual_indices: np.ndarray) -> float:
    if len(actual_indices) == 0:
        return float("nan")
    n_classes = prob_matrix.shape[1]
    one_hot = np.zeros_like(prob_matrix)
    one_hot[np.arange(len(actual_indices)), actual_indices] = 1.0
    return float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1)))


def _prob_matrix(df: pd.DataFrame, prefix: str, classes: list[str]) -> np.ndarray:
    cols = [f"{prefix}{cls}" for cls in classes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 1.0 / len(cols)
    return df[cols].to_numpy(dtype=float)


def generate_diagnostics(paths: WorkflowPaths, classes: Iterable[str]) -> None:
    outputs_dir = paths.outputs_dir
    predictions_path = outputs_dir / "predictions.csv"
    offline_path = outputs_dir / "offline_predictions.csv"
    ensemble_path = outputs_dir / "ensemble_predictions.csv"

    class_list = list(classes)

    if not predictions_path.exists():
        return

    predictions = pd.read_csv(predictions_path, parse_dates=["match_date"])
    offline = pd.read_csv(offline_path, parse_dates=["match_date"]) if offline_path.exists() else None
    ensemble = pd.read_csv(ensemble_path, parse_dates=["match_date"]) if ensemble_path.exists() else None

    report_lines: list[str] = []
    report_lines.append("# Model Diagnostics")

    for name, df, prefix in [
        ("Online", predictions, "model_prob_"),
        ("Offline", offline, "pred_prob_"),
        ("Ensemble", ensemble, "ensemble_prob_"),
    ]:
        if df is None:
            continue
        report_lines.append(f"\n## {name} Model")
        if "actual" not in df.columns:
            continue
        valid = df[df["actual"].notna()].copy()
        if valid.empty:
            report_lines.append("No evaluated predictions available.")
            continue
        prob_matrix = _prob_matrix(valid, prefix, class_list)
        actual_indices = valid["actual"].map(lambda cls: class_list.index(cls)).to_numpy()
        pred_col = "ensemble_prediction" if "ensemble_prediction" in valid.columns else "predicted"
        report_lines.append(
            f"Accuracy: {_safe_accuracy(valid[pred_col], valid['actual']):.3f}"
        )
        try:
            report_lines.append(f"Log loss: {log_loss(actual_indices, prob_matrix, labels=list(range(len(class_list)))):.3f}")
        except ValueError:
            report_lines.append("Log loss: unavailable (probabilities invalid)")
        try:
            report_lines.append(f"Brier score: {_safe_brier(prob_matrix, actual_indices):.3f}")
        except ValueError:
            report_lines.append("Brier score: unavailable")

        by_season = []
        for season, group in valid.groupby("season"):
            prob_season = _prob_matrix(group, prefix, class_list)
            actual_idxs = group["actual"].map(lambda cls: class_list.index(cls)).to_numpy()
            season_pred_col = "ensemble_prediction" if "ensemble_prediction" in group.columns else "predicted"
            acc = _safe_accuracy(group[season_pred_col], group["actual"])
            try:
                season_logloss = log_loss(actual_idxs, prob_season, labels=list(range(len(class_list))))
            except ValueError:
                season_logloss = float('nan')
            by_season.append((season, acc, season_logloss))
        report_lines.append("\nSeason breakdown (accuracy / logloss):")
        for season, acc, ll in by_season:
            report_lines.append(f"- {season}: {acc:.3f} / {ll:.3f}")

    report_path = outputs_dir / "diagnostics_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
