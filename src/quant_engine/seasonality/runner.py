"""Top-level orchestration for seasonality studies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore

from ..api.schemas import SeasonalitySpec
from ..backtest import engine
from ..core.dataset import load_ohlcv
from ..core.features import atr
from ..io import artifacts, ids
from ..signals.seasonality_signal import make_seasonality_signals
from ..validate import splitter
from ..persistence import db
from ..persistence.repo import (
    SeasonalityProfilesRepository,
    SeasonalityRunsRepository,
)
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


def _profiles_to_records(
    profiles_df: "pl.DataFrame",
    cfg: spec.NormalisedSeasonalitySpec,
) -> List[Dict[str, Any]]:
    """Convert the best profiles dataframe into persistence-ready rows."""

    if profiles_df.is_empty():
        return []

    measure = cfg.profile.measure
    timeframe = cfg.timeframe
    start = cfg.start.date().isoformat()
    end = cfg.end.date().isoformat()
    spec_id = cfg.persistence.spec_id
    dataset_id = cfg.persistence.dataset_id

    records: List[Dict[str, Any]] = []
    for row in profiles_df.to_dicts():
        timeframe_value = row.get("timeframe") or timeframe
        bin_value = row.get("bin")
        if isinstance(bin_value, bool):
            stored_bin = int(bin_value)
        elif isinstance(bin_value, (int, float)) and bin_value is not None:
            stored_bin = int(bin_value)
        elif bin_value is None:
            stored_bin = None
        else:
            stored_bin = str(bin_value)
        score_value = row.get("p_hat") if measure == "direction" else row.get("ret_mean")
        baseline_value = row.get("baseline")
        lift_value = row.get("lift")
        n_value = row.get("n")
        metrics_payload: Dict[str, Any] = {}
        for key in compute.CONDITIONAL_METRIC_NAMES:
            value = row.get(key)
            if value is None:
                continue
            if key == "n_runs":
                metrics_payload[key] = int(value)
            else:
                metrics_payload[key] = float(value)
        record = {
            "symbol": row.get("symbol"),
            "timeframe": timeframe_value,
            "dim": row.get("dim"),
            "bin": stored_bin,
            "measure": measure,
            "score": float(score_value) if score_value is not None else None,
            "n": int(n_value) if n_value is not None else None,
            "baseline": float(baseline_value) if baseline_value is not None else None,
            "lift": float(lift_value) if lift_value is not None else None,
            "start": start,
            "end": end,
            "spec_id": spec_id,
            "dataset_id": dataset_id,
        }
        record["metrics"] = metrics_payload
        records.append(record)
    return records


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
    df_source = load_ohlcv(spec_model.data)
    rows: List[Dict[str, Any]] = []
    for record in df_source.to_dict("records"):
        rec = dict(record)
        ts = rec.pop("ts", None)
        if ts is not None:
            ts_value = pd.to_datetime(ts, utc=True)
            rec["timestamp"] = ts_value.isoformat()
        rows.append(rec)
    rows.sort(key=lambda r: r.get("timestamp", ""))

    artifact_root: Path | None = None
    if cfg.artifacts.out_dir:
        artifact_root = Path(cfg.artifacts.out_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)

    run_id: str | None = None
    if cfg.persistence.enabled:
        run_id = ids.generate_id()
        with db.session() as conn:
            runs_repo = SeasonalityRunsRepository(conn)
            runs_repo.create(
                run_id,
                spec_id=cfg.persistence.spec_id,
                dataset_id=cfg.persistence.dataset_id,
                out_dir=str(artifact_root) if artifact_root is not None else None,
                status="running",
            )

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
    best_profiles_df: "pl.DataFrame" | None = None
    fold_summaries: List[Dict[str, Any]] = []
    atr_mult, r_mult = _atr_settings(cfg.tp_sl)
    profiles_records: List[Dict[str, Any]] = []
    best_summary_for_db: Dict[str, Any] | None = None
    result_payload: Dict[str, Any] = {}
    run_status = "completed"

    try:
        if not rows:
            result_payload = {
                "best_metrics": {},
                "active_bins": {},
                "artifacts": {},
                "folds": [],
            }
            best_summary_for_db = dict(result_payload)
        else:
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
                        or metrics.get("sharpe", 0.0)
                        > best_result.metrics.get("sharpe", 0.0)
                    )
                )

                profiles_path = (
                    fold_dir / "seasonality_profiles.parquet"
                    if fold_dir is not None
                    else None
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
                    best_profiles_df = profiles_df.clone()

            if best_result is None:
                result_payload = {
                    "best_metrics": {},
                    "active_bins": {},
                    "artifacts": {},
                    "folds": fold_summaries,
                }
                best_summary_for_db = dict(result_payload)
            else:
                artifacts_info: Dict[str, Any] = {}
                if (
                    best_result.profiles_path is not None
                    and best_result.profiles_path.exists()
                ):
                    artifacts_info["profiles"] = str(best_result.profiles_path)
                if best_result.summary_path is not None:
                    artifacts_info["summary"] = str(best_result.summary_path)
                if best_result.trades_path is not None:
                    artifacts_info["trades"] = str(best_result.trades_path)
                if best_result.equity_path is not None:
                    artifacts_info["equity"] = str(best_result.equity_path)

                result_payload = {
                    "best_metrics": best_result.metrics,
                    "active_bins": best_result.rules.to_serialisable()["active_bins"],
                    "rules_metadata": _default_rules_metadata(best_result.rules),
                    "artifacts": artifacts_info,
                    "folds": fold_summaries,
                }
                best_summary_for_db = dict(result_payload)
                if cfg.persistence.enabled and best_profiles_df is not None:
                    profiles_records = _profiles_to_records(best_profiles_df, cfg)
    except Exception:
        run_status = "failed"
        raise
    finally:
        if cfg.persistence.enabled and run_id is not None:
            with db.session() as conn:
                runs_repo = SeasonalityRunsRepository(conn)
                runs_repo.finish(
                    run_id,
                    "completed" if run_status == "completed" else "failed",
                    best_summary_for_db,
                )
                if run_status == "completed" and profiles_records:
                    profiles_repo = SeasonalityProfilesRepository(conn)
                    profiles_repo.bulk_upsert(profiles_records)

    if run_id is not None:
        result_payload = dict(result_payload)
        result_payload["run_id"] = run_id

    return result_payload
