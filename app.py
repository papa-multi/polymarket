"""
Textual-based CLI app exposing buttons to download data, run the model, and
inspect the latest metrics.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Static

from .workflow import (
    WorkflowPaths,
    collect_feature_columns,
    default_config,
    prepare_dataset,
    run_full_workflow,
    seasons_from_config,
)


class StatusLog(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer: list[str] = []

    def append(self, message: str) -> None:
        self._buffer.append(message)
        text = "\n".join(self._buffer[-200:])
        self.update(text)
        self.scroll_end(animate=False)


class FootballModelApp(App):
    CSS_PATH = None

    running: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static("Premier League Online Model", classes="title")
            with Horizontal():
                yield Button("Download & Run Model", id="run", variant="success")
                yield Button("Show Metrics", id="metrics", variant="primary")
            yield Static("Status", classes="section-title")
            self.log_widget = StatusLog()
            yield self.log_widget
        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if self.running:
            self.log_widget.append("Another task is running. Please wait.")
            return

        if event.button.id == "run":
            await self.run_model_workflow()
        elif event.button.id == "metrics":
            await self.show_metrics()

    async def run_model_workflow(self) -> None:
        self.running = True
        self.log_widget.append("Starting download and model run...")
        try:
            outputs = await asyncio.to_thread(self._run_pipeline_sync)
            summary = outputs["summary"]
            self.log_widget.append(
                f"Workflow complete. Model accuracy={summary.loc[summary['metric']=='model_overall_accuracy','value'].iloc[0]:.3f}"
            )
        except Exception as exc:
            self.log_widget.append(f"Error: {exc}")
        finally:
            self.running = False

    def _run_pipeline_sync(self):
        root = Path(__file__).resolve().parent
        config = default_config()
        paths = WorkflowPaths.default(root)
        seasons = seasons_from_config(config)
        dataset_path = prepare_dataset(paths, seasons)
        feature_columns = collect_feature_columns(dataset_path)
        return run_full_workflow(dataset_path, paths, feature_columns, config)

    async def show_metrics(self) -> None:
        root = Path(__file__).resolve().parent
        metrics_path = root / "outputs" / "season_metrics.csv"
        if not metrics_path.exists():
            self.log_widget.append("No metrics found. Run the model first.")
            return
        try:
            metrics = await asyncio.to_thread(pd.read_csv, metrics_path)
        except Exception as exc:
            self.log_widget.append(f"Failed to load metrics: {exc}")
            return
        self.log_widget.append(metrics.to_string(index=False))


def main() -> None:
    app = FootballModelApp()
    app.run()


if __name__ == "__main__":
    main()
