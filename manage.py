#!/usr/bin/env python3
"""
Convenience wrapper to run the interactive manager from inside the project directory.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_parent_on_path() -> None:
    parent = Path(__file__).resolve().parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


def main() -> None:
    _ensure_parent_on_path()
    from football_weight_model.cli import (
        install_prerequisites,
        download_data,
        build_dataset,
        run_model,
        run_online_only,
        train_offline_only,
        view_results,
        run_final_model,
        show_upcoming_predictions,
    )

    actions = {
        "1": ("Install prerequisites", install_prerequisites),
        "2": ("Run model (fixture pipeline; no downloads)", run_model),
        "3": ("View latest results", view_results),
        "4": ("Clean & refresh all data sources", download_data),
        "5": ("Build unified dataset", build_dataset),
        "6": ("Run online model only", run_online_only),
        "7": ("Train offline model only", train_offline_only),
        "8": ("Run final blended model", run_final_model),
        "9": ("Show upcoming week predictions", show_upcoming_predictions),
        "q": ("Quit", None),
    }

    while True:
        print("\nFootball Prediction Manager")
        for key, (label, _) in actions.items():
            print(f"[{key}] {label}")
        choice = input("Select an option: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break

        action = actions.get(choice)
        if not action:
            print("Invalid selection. Try again.")
            continue

        label, func = action
        if func is None:
            print("No action assigned.")
            continue

        print(f"\n=== {label} ===")
        try:
            func()
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
