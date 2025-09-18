"""Command line interface entry points."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, parse, request

import typer

from ..core import spec as spec_module

try:  # Job manager may not be available in lightweight tests
    from ..optimize.job_manager import JobManager  # type: ignore
except Exception:  # pragma: no cover - fallback used when import fails
    JobManager = None  # type: ignore

app = typer.Typer()
runs_app = typer.Typer()
stats_app = typer.Typer()
seasonality_app = typer.Typer()
app.add_typer(runs_app, name="runs")
app.add_typer(stats_app, name="stats")
app.add_typer(seasonality_app, name="seasonality")


class RunStatus(str, Enum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


@app.command("run-local")
def run_local(
    spec: Path = typer.Option(..., "--spec", exists=True, file_okay=True, dir_okay=False)
) -> None:
    """Run a specification synchronously using the local job manager."""

    sp = spec_module.load_spec(spec)
    if JobManager is not None:
        result: Dict[str, Any] = JobManager.submit(sp, synchronous=True)
    else:  # pragma: no cover - fallback for environments without job manager
        from ..optimize import runner

        result = runner.run(sp)
    best = result.get("best") or {}
    summary: Dict[str, Any] = {}
    if "metrics" in best:
        summary["metrics"] = best["metrics"]
    if "params" in best:
        summary["params"] = best["params"]
    typer.echo(json.dumps(summary, separators=(",", ":")))


@app.command("submit")
def submit(
    spec: Path = typer.Option(..., "--spec", exists=True, file_okay=True, dir_okay=False)
) -> None:
    """Submit a spec to the HTTP API and print the returned run identifier."""

    sp = spec_module.load_spec(spec)
    data = json.dumps(sp.model_dump()).encode()
    req = request.Request(
        "http://127.0.0.1:8000/submit", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with request.urlopen(req) as resp:
            if resp.status != 200:
                typer.echo(f"HTTP {resp.status}: {resp.reason}")
                raise typer.Exit(1)
            payload = json.loads(resp.read().decode())
    except error.HTTPError as e:
        typer.echo(f"HTTP {e.code}: {e.reason}")
        raise typer.Exit(1)
    except error.URLError as e:
        typer.echo(f"Connection error: {e.reason}")
        raise typer.Exit(1)
    typer.echo(payload.get("id", ""))


@stats_app.command("run")
def stats_run(
    spec: Path = typer.Option(..., "--spec", exists=True, file_okay=True, dir_okay=False)
) -> None:
    """Execute a statistics specification locally and display top entries."""

    from ..api.schemas import StatsSpec  # local import to keep startup light
    from ..stats.runner import run_stats
    import pandas as pd

    sp = StatsSpec.model_validate_json(Path(spec).read_text())
    df = run_stats(sp)
    if "split" in df.columns and "test" in df["split"].values:
        df = df[df["split"] == "test"]
    top = df.sort_values("lift", ascending=False).head(10)
    if top.empty:
        typer.echo("No statistics available")
    else:
        typer.echo(top.to_string(index=False))


@stats_app.command("show")
def stats_show(
    symbol: str = typer.Option(..., "--symbol"),
    event: str = typer.Option(..., "--event"),
    target: str = typer.Option(..., "--target"),
    timeframe: Optional[str] = typer.Option(None, "--timeframe"),
    limit: int = typer.Option(20, "--limit"),
    method: str = typer.Option("freq", "--method", help="freq or bayes"),
    significant_only: bool = typer.Option(
        False, "--significant-only/--no-significant-only"
    ),
) -> None:
    """Fetch persisted stats from the HTTP API and display them."""

    import pandas as pd

    params = {
        "symbol": symbol,
        "event": event,
        "target": target,
        "page": 1,
        "page_size": limit,
        "method": method,
        "significant_only": str(significant_only).lower(),
    }
    if timeframe is not None:
        params["timeframe"] = timeframe
    url = "http://127.0.0.1:8000/stats?" + parse.urlencode(params)
    try:
        with request.urlopen(url) as resp:
            if resp.status != 200:
                typer.echo(f"HTTP {resp.status}: {resp.reason}")
                raise typer.Exit(1)
            rows = json.loads(resp.read().decode())
    except error.HTTPError as e:
        typer.echo(f"HTTP {e.code}: {e.reason}")
        raise typer.Exit(1)
    except error.URLError as e:
        typer.echo(f"Connection error: {e.reason}")
        raise typer.Exit(1)
    df = pd.DataFrame(rows)
    if df.empty:
        typer.echo("No stats found")
    else:
        if method == "bayes":
            cols = ["p_mean", "hdi_low", "hdi_high", "lift_bayes"]
        else:
            cols = ["p_hat", "ci_low", "ci_high", "lift_freq"]
        display_cols = [
            "event",
            "condition_name",
            "condition_value",
            "target",
            "n",
            "successes",
            *[c for c in cols if c in df.columns],
        ]
        if "q_value" in df.columns:
            display_cols.append("q_value")
        typer.echo(df[display_cols].to_string(index=False))


@seasonality_app.command("run")
def seasonality_run(
    spec_path: Path = typer.Option(..., "--spec", exists=True, file_okay=True, dir_okay=False)
) -> None:
    """Execute a seasonality specification locally and display the summary."""

    from ..api.schemas import SeasonalitySpec
    from ..seasonality.runner import run as run_seasonality

    spec_model = SeasonalitySpec.model_validate_json(spec_path.read_text())
    result = run_seasonality(spec_model)
    summary = result.get("summary", {})
    typer.echo(json.dumps(summary, separators=(",", ":")))


@runs_app.command("list")
def list_runs(
    status: Optional[RunStatus] = typer.Option(None, "--status"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List runs from the HTTP API."""

    params = {"page": 1, "page_size": limit}
    if status is not None:
        params["status"] = status.value
    url = "http://127.0.0.1:8000/runs?" + parse.urlencode(params)
    try:
        with request.urlopen(url) as resp:
            if resp.status != 200:
                typer.echo(f"HTTP {resp.status}: {resp.reason}")
                raise typer.Exit(1)
            runs = json.loads(resp.read().decode())
    except error.HTTPError as e:
        typer.echo(f"HTTP {e.code}: {e.reason}")
        raise typer.Exit(1)
    except error.URLError as e:
        typer.echo(f"Connection error: {e.reason}")
        raise typer.Exit(1)
    header = "run_id | status | objective | started_at | finished_at"
    typer.echo(header)
    for r in runs:
        typer.echo(
            f"{r.get('run_id')} | {r.get('status')} | {r.get('objective')} | {r.get('started_at')} | {r.get('finished_at')}"
        )


@runs_app.command("show")
def show_run(run_id: str) -> None:
    """Show details about a specific run."""

    url = f"http://127.0.0.1:8000/runs/{run_id}"
    try:
        with request.urlopen(url) as resp:
            if resp.status != 200:
                typer.echo(f"HTTP {resp.status}: {resp.reason}")
                raise typer.Exit(1)
            detail = json.loads(resp.read().decode())
    except error.HTTPError as e:
        typer.echo(f"HTTP {e.code}: {e.reason}")
        raise typer.Exit(1)
    except error.URLError as e:
        typer.echo(f"Connection error: {e.reason}")
        raise typer.Exit(1)

    run = detail.get("run", {})
    metrics = detail.get("metrics", {}).get("aggregated")
    typer.echo(f"status: {run.get('status')}")
    typer.echo(f"objective: {run.get('objective')}")
    if metrics is not None:
        typer.echo("metrics: " + json.dumps(metrics, separators=(",", ":")))
    typer.echo(f"out_dir: {run.get('out_dir')}")
    best_params = run.get("best_params") or detail.get("best_params")
    if best_params:
        typer.echo("best_params: " + json.dumps(best_params, separators=(",", ":")))


if __name__ == "__main__":
    app()
