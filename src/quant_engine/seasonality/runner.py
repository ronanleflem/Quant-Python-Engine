"""Top-level orchestration for seasonality studies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore

from ..api.schemas import SeasonalitySpec
from ..backtest import engine
from ..core import dataset as core_dataset
from ..core.features import atr
from ..core.spec import DataSpec
from ..io import artifacts
from ..signals.seasonality_signal import make_seasonality_signals
from ..validate import splitter
from . import compute, profiles, spec


def _require_polars() -> None:
    if pl is None:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError("polars is required for seasonality runs")


def _rows_to_polars(rows: Sequence[Dict[str, Any]]) -> pl.DataFrame:
    """Return a Polars dataframe from the list-based dataset representation."""

    _require_polars()
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    if "timestamp" in df.columns:
        ts_dtype = df.schema.get("timestamp")
        if ts_dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, strict=False)
            )
        elif ts_dtype != pl.Datetime:
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))
    return df


def _default_rules_metadata(rules: profiles.SeasonalityRules) -> Dict[str, Any]:
    meta = dict(rules.metadata)
    meta.setdefault("thresholds", {})
    meta.setdefault("counts", {dim: len(bins) for dim, bins in rules.active_bins.items()})
    return meta


def _compute_profiles(
    dataset: pl.DataFrame,
    cfg: spec.NormalisedSeasonalitySpec,
    fold_dir: Path | None,
) -> pl.DataFrame:
    horizon_features = compute.prepare_features(dataset, cfg.profile)
    profiles_path = str(fold_dir) if fold_dir is not None else None
    return compute.compute_profiles(
        horizon_features,
        cfg.profile,
        timeframe=cfg.timeframe,
        period_start=None,
        period_end=None,
        artifacts_out_dir=profiles_path,
    )


def _rules_from_profiles(
    profiles_df: pl.DataFrame,
    cfg: spec.NormalisedSeasonalitySpec,
) -> profiles.SeasonalityRules:
    return profiles.select_bins(
        profiles_df,
        method=cfg.signal.method,
        threshold=cfg.signal.threshold,
        topk=cfg.signal.topk,
        dims=cfg.signal.dims,
        measure=cfg.profile.measure,
        combine=cfg.signal.combine,
    )


@dataclass
class FoldResult:
    index: int
    metrics: Dict[str, Any]
    rules: profiles.SeasonalityRules
    profiles_path: Path | None
    summary_path: Path | None
    trades_path: Path | None
    equity_path: Path | None


def _build_signals(
    rows: List[Dict[str, Any]],
    rules: profiles.SeasonalityRules,
) -> List[int]:
    df = _rows_to_polars(rows)
    df = compute.add_time_bins(df)
    df = make_seasonality_signals(df, rules)
    return [1 if bool(v) else 0 for v in df.get_column("long").to_list()]


def _atr_settings(tp_sl) -> tuple[float, float]:
    atr_mult = float(tp_sl.stop_loss) if tp_sl.stop_loss is not None else 1.0
    r_mult = float(tp_sl.take_profit) if tp_sl.take_profit is not None else 1.0
    return atr_mult, r_mult


def run(spec_model: SeasonalitySpec) -> Dict[str, Any]:
    """Execute a seasonality workflow and return aggregated metrics."""

    _require_polars()
    cfg = spec.normalise(spec_model)
    data_spec = DataSpec(
        path=str(cfg.dataset_path),
        symbols=list(cfg.symbols),
        start=cfg.start.date(),
        end=cfg.end.date(),
    )
    rows = core_dataset.load_dataset(data_spec)

    if not rows:
        return {
            "best_metrics": {},
            "active_bins": {},
            "artifacts": {},
            "folds": [],
        }

    artifact_root: Path | None = None
    if cfg.artifacts.out_dir:
        artifact_root = Path(cfg.artifacts.out_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)

    if cfg.validation.folds > 1:
        folds = splitter.generate_folds(
            rows,
            cfg.validation.train_months,
            cfg.validation.test_months,
            cfg.validation.folds,
            cfg.validation.embargo_days,
        )
    else:
        folds = [{"train": rows, "test": rows}]

    if not folds:
        folds = [{"train": rows, "test": rows}]

    best_result: FoldResult | None = None
    fold_summaries: List[Dict[str, Any]] = []
    atr_mult, r_mult = _atr_settings(cfg.tp_sl)

    for idx, fold in enumerate(folds):
        train_rows = fold.get("train", [])
        test_rows = fold.get("test", [])
        if not train_rows or not test_rows:
            continue

        train_df = _rows_to_polars(train_rows)
        if train_df.is_empty():
            continue

        fold_dir = artifact_root / f"fold_{idx}" if artifact_root is not None else None
        if fold_dir is not None:
            fold_dir.mkdir(parents=True, exist_ok=True)

        profiles_df = _compute_profiles(train_df, cfg, fold_dir)
        if profiles_df.is_empty():
            continue

        rules = _rules_from_profiles(profiles_df, cfg)
        signals = _build_signals(test_rows, rules)
        if not signals:
            continue

        atr_values = atr.compute(test_rows)
        trades, equity, metrics = engine.run(
            test_rows,
            signals,
            atr_values,
            atr_mult,
            r_mult,
            cfg.execution.slippage_bps,
            cfg.execution.commission_bps,
        )

        metrics = dict(metrics)
        metrics["fold_index"] = idx
        metrics["n_trades"] = int(metrics.get("trades", 0))
        metrics["rules_counts"] = {
            dim: len(bins) for dim, bins in rules.active_bins.items()
        }
        fold_summary = {
            "fold": idx,
            "metrics": metrics,
            "rules": rules.to_serialisable(),
        }
        fold_summaries.append(fold_summary)

        meets_min_trades = metrics["n_trades"] >= cfg.validation.min_trades
        is_better = (
            meets_min_trades
            and (
                best_result is None
                or metrics.get("sharpe", 0.0) > best_result.metrics.get("sharpe", 0.0)
            )
        )

        profiles_path = (
            fold_dir / "seasonality_profiles.parquet" if fold_dir is not None else None
        )

        if is_better:
            summary_path = None
            trades_path = None
            equity_path = None
            if fold_dir is not None:
                summary_payload = {
                    "metrics": metrics,
                    "rules": rules.to_serialisable(),
                }
                summary_path = fold_dir / "summary.json"
                artifacts.write_summary(summary_path, summary_payload)
                trades_path = fold_dir / "trades.parquet"
                equity_path = fold_dir / "equity.parquet"
                artifacts.write_trades(trades_path, trades)
                artifacts.write_equity(equity_path, equity)
            best_result = FoldResult(
                index=idx,
                metrics=metrics,
                rules=rules,
                profiles_path=profiles_path,
                summary_path=summary_path,
                trades_path=trades_path,
                equity_path=equity_path,
            )

    if best_result is None:
        return {
            "best_metrics": {},
            "active_bins": {},
            "artifacts": {},
            "folds": fold_summaries,
        }

    artifacts_info: Dict[str, Any] = {}
    if best_result.profiles_path is not None and best_result.profiles_path.exists():
        artifacts_info["profiles"] = str(best_result.profiles_path)
    if best_result.summary_path is not None:
        artifacts_info["summary"] = str(best_result.summary_path)
    if best_result.trades_path is not None:
        artifacts_info["trades"] = str(best_result.trades_path)
    if best_result.equity_path is not None:
        artifacts_info["equity"] = str(best_result.equity_path)

    return {
        "best_metrics": best_result.metrics,
        "active_bins": best_result.rules.to_serialisable()["active_bins"],
        "rules_metadata": _default_rules_metadata(best_result.rules),
        "artifacts": artifacts_info,
        "folds": fold_summaries,
    }
