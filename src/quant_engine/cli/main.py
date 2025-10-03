"""Command line interface entry points."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
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
levels_app = typer.Typer()
app.add_typer(runs_app, name="runs")
app.add_typer(stats_app, name="stats")
app.add_typer(seasonality_app, name="seasonality")
app.add_typer(levels_app, name="levels")


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


@seasonality_app.command("optimize")
def seasonality_optimize(
    spec_path: Path = typer.Option(..., "--spec", exists=True, file_okay=True, dir_okay=False)
) -> None:
    """Run the seasonality Optuna optimisation loop locally."""

    from ..api.schemas import SeasonalitySpec
    from ..seasonality.optimize import run_optimization

    spec_model = SeasonalitySpec.model_validate_json(spec_path.read_text())
    result = run_optimization(spec_model)
    payload = {
        "best_value": result.get("best_value"),
        "best_params": result.get("best_params"),
        "best_metrics": result.get("best_metrics"),
        "paths": result.get("paths"),
    }
    typer.echo(json.dumps(payload, separators=(",", ":")))


@seasonality_app.command("profiles")
def seasonality_profiles(
    symbol: Optional[str] = typer.Option(None, "--symbol"),
    timeframe: Optional[str] = typer.Option(None, "--timeframe"),
    dim: Optional[str] = typer.Option(None, "--dim"),
    measure: Optional[str] = typer.Option(None, "--measure"),
    metrics: Optional[str] = typer.Option(
        None,
        "--metrics",
        help="Comma-separated conditional metrics required (e.g. run_len_up_mean,p_breakout_up)",
    ),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """Fetch persisted seasonality profiles from the HTTP API."""

    import pandas as pd

    params = {"page": 1, "page_size": limit}
    if symbol:
        params["symbol"] = symbol
    if timeframe:
        params["timeframe"] = timeframe
    if dim:
        params["dim"] = dim
    if measure:
        params["measure"] = measure
    if metrics:
        params["metrics"] = metrics
    url = "http://127.0.0.1:8000/seasonality/profiles?" + parse.urlencode(params)
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
        typer.echo("No seasonality profiles found")
        return
    display_cols = [
        "symbol",
        "timeframe",
        "dim",
        "bin",
        "measure",
        "score",
        "n",
        "baseline",
        "lift",
    ]
    metric_values = df.get("metrics")
    if metric_values is not None:
        metrics_df = pd.json_normalize(metric_values).fillna("")
        df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)
        metric_columns = [col for col in metrics_df.columns if col]
        display_cols.extend(metric_columns)
    present_cols = [col for col in display_cols if col in df.columns]
    typer.echo(df[present_cols].to_string(index=False))


@seasonality_app.command("compare")
def seasonality_compare(
    symbols: List[str] = typer.Option(
        ..., "--symbols", help="Deux symboles à comparer"
    ),
    dim: str = typer.Option(..., "--dim", help="Dimension à comparer (ex: hour)"),
    timeframe: Optional[str] = typer.Option(None, "--timeframe"),
    measure: str = typer.Option("direction", "--measure"),
) -> None:
    """Comparer les lifts saisonniers de deux symboles pour une dimension."""

    try:
        import polars as pl
    except ModuleNotFoundError:  # pragma: no cover - dependency optional
        typer.echo("polars est requis pour comparer les profils saisonniers")
        raise typer.Exit(1)

    from ..seasonality import profiles as profiles_module

    if len(symbols) != 2:
        typer.echo("Veuillez fournir exactement deux symboles via --symbols")
        raise typer.Exit(1)

    tables: list[pl.DataFrame] = []
    for symbol in symbols:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "dim": dim,
            "page": 1,
            "page_size": 1000,
            "measure": measure,
        }
        if timeframe is not None:
            params["timeframe"] = timeframe
        url = "http://127.0.0.1:8000/seasonality/profiles?" + parse.urlencode(params)
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
        table = pl.DataFrame(rows)
        if "metrics" in table.columns:
            table = table.drop("metrics")
        tables.append(table)

    if not tables or any(table.is_empty() for table in tables):
        typer.echo("Profils indisponibles pour l'une des séries")
        raise typer.Exit(1)

    comparison, corr = profiles_module.compare_profiles(tables[0], tables[1], dim)
    if comparison.is_empty():
        typer.echo("Aucune intersection de bins sur cette dimension")
        raise typer.Exit(0)

    if corr is not None:
        typer.echo(f"Corrélation des lifts: {corr:.4f}")
    else:
        typer.echo("Corrélation des lifts: N/A")

    try:
        import pandas as pd
    except ModuleNotFoundError:  # pragma: no cover - affichage dégradé
        typer.echo(comparison.to_string())
        return

    typer.echo(pd.DataFrame(comparison.to_dicts()).to_string(index=False))


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


@levels_app.command("build")
def levels_build(
    spec: Path = typer.Option(..., "--spec", exists=True, file_okay=True, dir_okay=False)
) -> None:
    """Compute and persist slow levels defined in a JSON specification."""

    from ..levels.schemas import LevelsBuildSpec
    from ..levels.runner import run_levels_build

    spec_model = LevelsBuildSpec.model_validate_json(Path(spec).read_text())
    result = run_levels_build(spec_model)
    typer.echo(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    app()
