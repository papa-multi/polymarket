"""
Command-line utility for managing the Premier League prediction workflow.

Options exposed:
1. Install prerequisites (Python packages)
2. Run the model using existing data
3. View the latest prediction metrics
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np

from .data_refresh import refresh_all_sources
from .workflow import (
    WorkflowPaths,
    default_config,
    seasons_from_config,
)
from .fixture_features import build_unified_dataset as build_fixture_unified_dataset
from .pipeline_v3 import run_pipeline as run_fixture_pipeline
from .team_names import canonical_team_name

REQUIRED_PACKAGES: List[str] = [
    "pandas",
    "numpy",
    "requests",
    "pyyaml",
    "textual",
    "scikit-learn",
    "beautifulsoup4",
]


def _import_pandas():
    import importlib

    return importlib.import_module("pandas")


def install_prerequisites() -> None:
    """Install required Python packages into the current interpreter."""
    print("Installing prerequisites...")
    in_virtualenv = (
        hasattr(sys, "base_prefix")
        and sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    ) or "VIRTUAL_ENV" in os.environ

    for package in REQUIRED_PACKAGES:
        print(f" - {package}")
        cmd = [sys.executable, "-m", "pip", "install", package]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            if in_virtualenv:
                raise
            fallback_cmd = cmd + ["--break-system-packages"]
            print("   (retrying with --break-system-packages)")
            subprocess.run(fallback_cmd, check=True)
    print("All prerequisites installed.")


def _prepare_project_context():
    project_root = Path(__file__).resolve().parent
    config = default_config()
    paths = WorkflowPaths.default(project_root)
    seasons = seasons_from_config(config)
    return project_root, config, paths, seasons


def _write_upcoming_fixtures(days: int) -> bool:
    """Harvest upcoming fixtures from football-data raw files and write schedule CSV."""
    pd = _import_pandas()
    project_root, config, paths, seasons = _prepare_project_context()
    understat_dir = project_root / "football_weight_model" / "data" / "raw" / "understat"

    today = pd.Timestamp.today().normalize()
    horizon = today + pd.Timedelta(days=days)

    fixtures: list[dict] = []
    for season in seasons:
        raw_file = paths.raw_dir / f"epl_{season.code}.csv"
        if not raw_file.exists():
            continue

        try:
            raw_df = pd.read_csv(raw_file)
        except Exception:
            continue

        if "Date" not in raw_df.columns or "HomeTeam" not in raw_df.columns or "AwayTeam" not in raw_df.columns:
            continue

        raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce", dayfirst=True).dt.normalize()
        mask = raw_df["Date"].between(today, horizon)
        future = raw_df.loc[mask, ["Date", "HomeTeam", "AwayTeam"]].dropna()
        for _, row in future.iterrows():
            match_date = row["Date"]
            home = canonical_team_name(str(row["HomeTeam"]))
            away = canonical_team_name(str(row["AwayTeam"]))
            if pd.isna(home) or pd.isna(away):
                continue
            fixtures.append(
                {
                    "match_date": match_date.strftime("%Y-%m-%d"),
                    "home_team": home,
                    "away_team": away,
                }
            )

    if not fixtures and understat_dir.exists():
        for path in sorted(understat_dir.glob("matches_EPL_*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            for match in data:
                raw_goals = match.get("goals", {})
                home_goals = raw_goals.get("h")
                away_goals = raw_goals.get("a")
                if home_goals not in (None, "", "null") and away_goals not in (None, "", "null"):
                    continue
                match_dt = pd.to_datetime(match.get("datetime"), errors="coerce")
                if pd.isna(match_dt):
                    continue
                match_dt = match_dt.tz_localize(None).normalize()
                if not today <= match_dt <= horizon:
                    continue
                home = canonical_team_name(match["h"]["title"])
                away = canonical_team_name(match["a"]["title"])
                if pd.isna(home) or pd.isna(away):
                    continue
                fixtures.append(
                    {
                        "match_date": match_dt.strftime("%Y-%m-%d"),
                        "home_team": home,
                        "away_team": away,
                    }
                )

    if not fixtures:
        return False

    upcoming_path = paths.raw_dir / "upcoming_fixtures.csv"
    new_df = pd.DataFrame(fixtures).drop_duplicates()
    if upcoming_path.exists():
        try:
            existing = pd.read_csv(upcoming_path, parse_dates=["match_date"])
        except Exception:
            existing = pd.DataFrame(columns=new_df.columns)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined["match_date"] = pd.to_datetime(combined["match_date"], errors="coerce")
        combined = (
            combined.dropna(subset=["match_date", "home_team", "away_team"])
            .drop_duplicates()
            .sort_values("match_date")
        )
    else:
        combined = new_df.sort_values("match_date")

    combined.to_csv(upcoming_path, index=False)
    return True


def _ensure_upcoming_predictions(days: int) -> None:
    """Guarantee forecasts exist for the next `days` fixtures (runs build + model if needed)."""
    _, _, paths, _ = _prepare_project_context()
    dataset_path = paths.processed_dir / "epl_unified_team_matches.csv"

    # Generate schedule if required
    wrote_any = _write_upcoming_fixtures(days)
    if not wrote_any and dataset_path.exists():
        # No new fixtures discovered; nothing to do.
        return

    # Rebuild unified dataset to include the schedule and rerun pipeline.
    result = build_fixture_unified_dataset()
    run_fixture_pipeline(result.dataset_path, paths.outputs_dir)


RESULT_ORDER = ["win", "draw", "loss"]
_LABEL_TO_INDEX = {cls: idx for idx, cls in enumerate(RESULT_ORDER)}


def _brier_from_predictions(df) -> float:
    if df.empty:
        return float("nan")
    probs = df[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy(dtype=float)
    labels = df["label"].map(_LABEL_TO_INDEX).to_numpy()
    valid_mask = ~np.isnan(labels)
    if not valid_mask.any():
        return float("nan")
    probs = probs[valid_mask]
    labels = labels[valid_mask].astype(int)
    eye = np.eye(len(RESULT_ORDER), dtype=float)
    one_hot = eye[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _season_metrics_from_predictions(df):
    if df is None or df.empty:
        return None
    pd = _import_pandas()
    seasons = sorted(df["season"].unique())
    rows = []
    prev_accuracy = None
    for season in seasons:
        season_df = df[df["season"] == season]
        accuracy = float(season_df["correct"].mean()) if not season_df.empty else float("nan")
        brier = _brier_from_predictions(season_df)
        change = accuracy - prev_accuracy if prev_accuracy is not None else np.nan
        rows.append(
            {
                "season": season,
                "model_accuracy": accuracy,
                "model_brier_score": brier,
                "accuracy_change_from_previous": change,
            }
        )
        prev_accuracy = accuracy
    return pd.DataFrame(rows)


def download_data() -> None:
    """Clean existing artefacts, download all sources, and rebuild the unified dataset."""
    print("Cleaning existing data and downloading all sources...")
    refresh_all_sources(delete_existing=True)
    print("Raw sources refreshed. Rebuilding unified dataset...")
    result = build_fixture_unified_dataset()
    print(f"Finished. Unified dataset saved to {result.dataset_path}")
    print(f"Data quality summary written to {result.quality_report_path}")
    if result.issues_path and result.issues_path.exists():
        print(f"Diagnostics written to {result.issues_path}")


def run_model() -> None:
    """Train and evaluate the fixture-level model (same logic as option [8])."""
    print("Running fixture modelling pipeline...")
    _, config, paths, _ = _prepare_project_context()

    dataset_path = paths.processed_dir / "epl_unified_team_matches.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Unified dataset missing. Run option [4] to refresh data before running the model."
        )

    results = run_fixture_pipeline(dataset_path, paths.outputs_dir)
    summary = results["summary"]

    print("Fixture model complete. Summary:")
    print(summary.to_string(index=False))


def run_final_model() -> None:
    """Alias for run_model (kept for backward compatibility)."""
    run_model()


def train_model() -> None:
    """Explicit training entrypoint (same as run_model)."""
    run_model()


def show_upcoming_predictions(days: int = 7, count: Optional[int] = None) -> None:
    """Display win/draw/loss probabilities for upcoming fixtures."""
    pd = _import_pandas()
    _, _, paths, _ = _prepare_project_context()

    fixture_file = paths.outputs_dir / "forecast_clean.csv"
    refresh_attempted = False

    def _load_forecast() -> Optional[pd.DataFrame]:
        if not fixture_file.exists():
            return None
        return pd.read_csv(fixture_file, parse_dates=["match_date"])

    df = _load_forecast()
    if df is None or df.empty or (
        "home_goals" in df.columns and df["home_goals"].isna().sum() == 0
    ):
        _ensure_upcoming_predictions(days)
        refresh_attempted = True
        df = _load_forecast()

    if df is None:
        print("❗ Unable to locate final fixture predictions. Try running option [2] first.")
        return

    if "home_goals" in df.columns:
        df = df[df["home_goals"].isna()].copy()
    elif "actual" in df.columns:
        df = df[df["actual"].isna()].copy()

    if len(df) < 5 and not refresh_attempted:
        _ensure_upcoming_predictions(days)
        refresh_attempted = True
        df = _load_forecast()
        if df is not None:
            if "home_goals" in df.columns:
                df = df[df["home_goals"].isna()].copy()
            elif "actual" in df.columns:
                df = df[df["actual"].isna()].copy()

    if df.empty and not refresh_attempted:
        _ensure_upcoming_predictions(days)
        df = _load_forecast()
        if df is None:
            print("❗ Unable to locate final fixture predictions after refresh.")
            return
        if "home_goals" in df.columns:
            df = df[df["home_goals"].isna()].copy()
        elif "actual" in df.columns:
            df = df[df["actual"].isna()].copy()

    if df.empty:
        print("ℹ️ No fixtures without results found in predictions.")
        return

    today = date.today()
    end_date = today + timedelta(days=days)
    mask = df["match_date"].dt.date.between(today, end_date)
    upcoming = df.loc[mask].sort_values("match_date")
    window_label = f"{today} to {end_date}"

    if upcoming.empty:
        earliest = df["match_date"].min()
        if pd.isna(earliest):
            print("ℹ️ No future fixtures available.")
            return
        fallback_end = earliest + timedelta(days=days)
        upcoming = df[(df["match_date"] >= earliest) & (df["match_date"] <= fallback_end)].sort_values("match_date")
        if upcoming.empty:
            upcoming = df.sort_values("match_date")
        window_label = f"{earliest.date()} onward (no fixtures inside {days} days from today)"
        if earliest.date() > end_date:
            print(
                f"ℹ️ No fixtures scheduled between {today} and {end_date}. "
                f"Showing matches starting from {earliest.date()} instead."
            )

    # Drop mirrored duplicates (home/away pair showing twice)
    pair_key = upcoming.apply(lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1)
    upcoming = upcoming.loc[~pair_key.duplicated()].copy()

    if count is not None and count > 0:
        upcoming = upcoming.head(count)

    print(f"Upcoming fixtures ({window_label}):")
    print(
        upcoming[
            [
                "match_date",
                "home_team",
                "away_team",
                "final_p_home",
                "final_p_draw",
                "final_p_away",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )


def build_dataset() -> None:
    """Rebuild the unified dataset from existing raw files."""
    print("Building unified dataset from current raw data...")
    result = build_fixture_unified_dataset()
    print(f"Unified dataset written to {result.dataset_path}")
    print(f"Data quality summary written to {result.quality_report_path}")
    if result.issues_path and result.issues_path.exists():
        print(f"Diagnostics written to {result.issues_path}")


def run_online_only() -> None:
    """Deprecated placeholder for legacy command."""
    print("Online-only model is deprecated in the new pipeline. Use option [2] or [8].")


def train_offline_only() -> None:
    """Deprecated placeholder for legacy command."""
    print("Offline-only model is deprecated in the new pipeline. Use option [2] or [8].")


def view_results() -> None:
    """Display the most recent metrics and prediction head/tail."""
    pd = _import_pandas()
    project_root = Path(__file__).resolve().parent
    outputs_dir = project_root / "outputs"
    summary_file = outputs_dir / "summary.csv"
    season_metrics_file = outputs_dir / "season_metrics.csv"
    validation_file = outputs_dir / "validation_metrics.csv"
    predictions_file = outputs_dir / "final_predictions.csv"
    forecast_file = outputs_dir / "forecast_clean.csv"

    summary = pd.read_csv(summary_file) if summary_file.exists() else None
    season_metrics = pd.read_csv(season_metrics_file) if season_metrics_file.exists() else None
    validation_metrics = pd.read_csv(validation_file) if validation_file.exists() else None
    predictions = pd.read_csv(predictions_file) if predictions_file.exists() else None

    print("=== Summary ===")
    if summary is not None:
        print(summary.to_string(index=False))
    else:
        print("Summary file missing.")

    print("\n=== Season Metrics ===")
    if season_metrics is not None and not season_metrics.empty:
        print(season_metrics.to_string(index=False))
    else:
        print("Season metrics file missing.")

    if validation_metrics is not None:
        print("\n=== Walk-forward Validation (per fold) ===")
        print(validation_metrics.to_string(index=False))

    if predictions is not None:
        print("\n=== Final Predictions (Last 10 Rows) ===")
        cols = [
            "season",
            "match_date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "label",
            "final_p_home",
            "final_p_draw",
            "final_p_away",
            "prediction",
            "correct",
        ]
        cols = [c for c in cols if c in predictions.columns]
        print(predictions[cols].tail(10).to_string(index=False))

    if forecast_file.exists():
        forecast = pd.read_csv(forecast_file, parse_dates=["match_date"])
        if not forecast.empty:
            print("\n=== Upcoming Fixtures (Next 10) ===")
            display = forecast[
                [
                    "match_date",
                    "home_team",
                    "away_team",
                    "final_p_home",
                    "final_p_draw",
                    "final_p_away",
                    "confidence",
                ]
            ].head(10)
            print(display.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Football model management CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("install", help="Install prerequisites.")
    subparsers.add_parser("download", help="Download/refresh data without running the model.")
    subparsers.add_parser("build", help="Build the unified dataset from current raw files.")
    subparsers.add_parser("run", help="Run the fixture-level model using existing data.")
    subparsers.add_parser("predict", help="Alias for run – generate predictions using existing data.")
    subparsers.add_parser("online", help="(Deprecated) Legacy online model placeholder.")
    subparsers.add_parser("train", help="Train the fixture model (same as run).")
    subparsers.add_parser("final", help="Alias for run – emit forecasts using the fixture model.")
    forecast_parser = subparsers.add_parser("forecast", help="Show predictions for upcoming fixtures.")
    forecast_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days ahead to include (default: 7).",
    )
    forecast_parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Limit the number of fixtures displayed (optional).",
    )
    subparsers.add_parser("view", help="Show latest metrics and predictions.")
    subparsers.add_parser("evaluate", help="Alias for view – display latest evaluation tables.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "install":
        install_prerequisites()
    elif args.command == "download":
        download_data()
    elif args.command == "build":
        build_dataset()
    elif args.command in {"run", "predict"}:
        run_model()
    elif args.command == "online":
        run_online_only()
    elif args.command == "train":
        train_model()
    elif args.command == "final":
        run_final_model()
    elif args.command == "forecast":
        show_upcoming_predictions(days=args.days, count=args.count)
    elif args.command in {"view", "evaluate"}:
        view_results()
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
