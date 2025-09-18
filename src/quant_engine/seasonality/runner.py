"""Top-level orchestration for seasonality studies."""
from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Dict, Any

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore

from ..api.schemas import SeasonalitySpec
from ..signals.seasonality_signal import build_seasonality_signal
from . import compute, profiles, spec


def _require_polars() -> None:
    if pl is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("polars is required for seasonality runs")


def _load_dataset(path: Path) -> pl.DataFrame:
    """Load a dataset from disk inferring the format from the extension."""

    _require_polars()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pl.read_csv(path, try_parse_dates=True)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _ensure_datetime(df: pl.DataFrame) -> pl.DataFrame:
    _require_polars()
    if df.is_empty():
        return df
    if df.schema.get("timestamp") == pl.Datetime:
        return df
    if df.schema.get("timestamp") == pl.Utf8:
        return df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, strict=False))
    return df.with_columns(pl.col("timestamp").cast(pl.Datetime))


def run(spec_model: SeasonalitySpec) -> Dict[str, Any]:
    """Execute a seasonality workflow and return the summary."""

    _require_polars()
    cfg = spec.normalise(spec_model)
    df = _load_dataset(cfg.dataset_path)
    df = _ensure_datetime(df)
    if "symbol" in df.columns:
        df = df.filter(pl.col("symbol").is_in(cfg.symbols))
    if not df.is_empty():
        df = df.filter(
            (pl.col("timestamp") >= cfg.start) & (pl.col("timestamp") <= cfg.end)
        )
    artifact_paths: list[str] = []
    if cfg.artifacts.out_dir:
        artifact_paths.append(str(Path(cfg.artifacts.out_dir)))

    if df.is_empty():
        return {
            "summary": {"n_rows": 0, "n_trades": 0, "best_bins": {}},
            "profiles": {},
            "signal": {"active_bins": {}},
            "artifacts": artifact_paths,
        }

    features = compute.prepare_features(df, cfg.profile)
    profiles_map = compute.compute_profiles(features, cfg.profile)
    active_bins = profiles.select_active_bins(profiles_map, cfg.signal, cfg.profile)
    signal_df = build_seasonality_signal(features, active_bins, cfg.signal.combine)

    trades = signal_df.filter(pl.col("long") == 1)
    returns = (
        trades.get_column("forward_ret")
        if "forward_ret" in trades.columns
        else pl.Series(name="forward_ret", values=[], dtype=pl.Float64)
    )
    returns = returns.drop_nulls()
    n_trades = int(trades.height)
    avg_return = float(returns.mean()) if returns.len() else 0.0
    std_return = float(returns.std()) if returns.len() > 1 else 0.0
    hit_rate = float((returns > 0).sum() / returns.len()) if returns.len() else 0.0
    sharpe = (avg_return / std_return * sqrt(252)) if std_return else 0.0

    summary = {
        "n_rows": int(df.height),
        "n_trades": n_trades,
        "avg_return": avg_return,
        "hit_rate": hit_rate,
        "sharpe": sharpe,
        "best_bins": {dim: list(bins) for dim, bins in active_bins.items()},
    }

    return {
        "summary": summary,
        "profiles": profiles.summarise_profiles(profiles_map),
        "signal": {"active_bins": summary["best_bins"]},
        "artifacts": artifact_paths,
    }
