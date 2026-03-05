"""Top-level orchestration for seasonality studies."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

try:  # pragma: no cover - optional dependency
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - used when dependency missing
    pl = None  # type: ignore

from ..api.schemas import SeasonalitySpec, SeasonalityProfileSpec
from ..backtest import engine
from ..core.dataset import load_ohlcv
from ..core.features import atr
from ..io import artifacts, ids
from ..signals.seasonality_signal import DIMENSION_TO_COLUMN, make_seasonality_signals
from ..market_intelligence.pipeline import compute_features as mi_compute_features, label_regimes as mi_label_regimes
from ..validate import splitter
from ..persistence import db
from ..persistence.repo import (
    SeasonalityProfilesRepository,
    SeasonalityRunsRepository,
)
from . import compute, profiles, spec

logger = logging.getLogger(__name__)


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
        df = _ensure_utc_timestamp(df, "timestamp")
    return df


def _ensure_utc_timestamp(dataset: pl.DataFrame, column: str) -> pl.DataFrame:
    """Normalise a Polars datetime column to timezone-aware UTC."""

    if column not in dataset.columns:
        return dataset

    dtype = dataset.schema.get(column)
    if dtype == pl.Utf8:
        expr = pl.col(column).str.to_datetime(strict=False, utc=True)
    elif isinstance(dtype, pl.Datetime):
        tz = getattr(dtype, "time_zone", None)
        if tz == "UTC":
            return dataset
        expr = (
            pl.col(column).dt.replace_time_zone("UTC")
            if tz is None
            else pl.col(column).dt.convert_time_zone("UTC")
        )
    else:
        expr = pl.col(column).cast(pl.Datetime).dt.replace_time_zone("UTC")
    return dataset.with_columns(expr.alias(column))


def _rows_to_pandas(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Return a pandas dataframe from the list-based dataset representation."""

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _add_time_bins_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Augment the dataset with calendar bins using pandas."""

    if df.empty:
        return df.copy()
    df = df.copy()
    if "timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = ts
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.weekday
    df["month"] = ts.dt.month
    df["month_of_year"] = ts.dt.month
    df["day_in_month"] = ts.dt.day
    df["week_in_month"] = ((df["day_in_month"] - 1) // 7 + 1).astype(int)
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)
    df["session"] = ts.apply(compute.assign_session)
    df["is_month_start"] = df["day_in_month"] == 1

    year = ts.dt.year
    month = ts.dt.month
    max_day = (
        df.assign(_year=year, _month=month)
        .groupby(["symbol", "_year", "_month"])["timestamp"]
        .transform(lambda s: s.dt.day.max())
    )
    days_from_end = max_day - df["day_in_month"]
    df["is_month_end"] = days_from_end == 0
    df["is_news_hour"] = df["hour"].isin([13, 14, 20])
    df["is_third_friday"] = ts.apply(compute._is_third_friday)
    for offset, name in enumerate(compute.LAST_DAY_COLUMNS, start=1):
        df[name] = days_from_end == offset
    for idx, name in enumerate(compute.MONTH_FLAG_COLUMNS, start=1):
        df[name] = df["month"] == idx

    if "roll_id" in df.columns:
        df["is_rollover_day"] = (
            df.groupby("symbol")["roll_id"].apply(lambda s: s.ne(s.shift(1))).fillna(False)
        )
    return df


def _prepare_features_pandas(
    dataset: pd.DataFrame,
    profile: SeasonalityProfileSpec,
) -> pd.DataFrame:
    """Prepare features for the pandas-backed seasonality workflow."""

    if dataset.empty:
        return dataset.copy()
    df = dataset.sort_values(["symbol", "timestamp"]).copy()
    horizon = max(int(profile.ret_horizon), 1)
    df = _add_time_bins_pandas(df)
    fwd = df.groupby("symbol")["close"].shift(-horizon)
    eps = 1e-12
    returns = (fwd / df["close"] - 1.0).where(
        df["close"].abs() > eps, other=pd.NA
    )
    return_col = f"return_h{horizon}"
    df[return_col] = returns
    df["forward_ret"] = returns
    direction_col = f"direction_h{horizon}"
    df[direction_col] = returns.gt(0).astype("Int64")
    df["direction"] = df[direction_col]
    return df


def _compute_profiles_pandas(
    dataset: pd.DataFrame,
    cfg: spec.NormalisedSeasonalitySpec,
    fold_dir: Path | None,
) -> pd.DataFrame:
    """Compute light-weight seasonality profiles using pandas."""

    if dataset.empty:
        return pd.DataFrame()
    horizon = max(int(cfg.profile.ret_horizon), 1)
    period_start = dataset["timestamp"].min() if "timestamp" in dataset.columns else None
    period_end = dataset["timestamp"].max() if "timestamp" in dataset.columns else None
    dims = list(compute.iter_active_bins(cfg.profile))

    tables: list[pd.DataFrame] = []
    for dim in dims:
        if dim not in dataset.columns:
            continue
        group_cols = ["symbol", dim]
        if cfg.profile.measure == "direction":
            direction_col = f"direction_h{horizon}"
            df_valid = dataset[dataset[direction_col].notna()]
            if df_valid.empty:
                continue
            grouped = (
                df_valid.groupby(group_cols)[direction_col]
                .agg(["count", "sum"])
                .rename(columns={"count": "n", "sum": "successes"})
            )
            baseline = df_valid.groupby("symbol")[direction_col].mean()
            grouped["baseline"] = grouped.index.get_level_values("symbol").map(baseline)
            grouped["p_hat"] = grouped["successes"] / grouped["n"]
            grouped["insufficient"] = grouped["n"] < cfg.profile.min_samples_bin
            grouped.loc[grouped["insufficient"], "p_hat"] = pd.NA
            grouped["lift"] = grouped["p_hat"] - grouped["baseline"]
            grouped.loc[grouped["insufficient"], "lift"] = pd.NA
            grouped["ci_low"] = pd.NA
            grouped["ci_high"] = pd.NA
            table = grouped.reset_index().rename(columns={dim: "bin"})
        else:
            return_col = f"return_h{horizon}"
            df_valid = dataset[dataset[return_col].notna()]
            if df_valid.empty:
                continue
            grouped = (
                df_valid.groupby(group_cols)[return_col]
                .agg(["count", "mean", "median", "std"])
                .rename(
                    columns={
                        "count": "n",
                        "mean": "ret_mean",
                        "median": "ret_median",
                        "std": "ret_std",
                    }
                )
            )
            baseline = df_valid.groupby("symbol")[return_col].mean()
            grouped["baseline"] = grouped.index.get_level_values("symbol").map(baseline)
            grouped["insufficient"] = grouped["n"] < cfg.profile.min_samples_bin
            grouped.loc[grouped["insufficient"], ["ret_mean", "ret_median", "ret_std"]] = pd.NA
            grouped["lift"] = grouped["ret_mean"] - grouped["baseline"]
            grouped.loc[grouped["insufficient"], "lift"] = pd.NA
            table = grouped.reset_index().rename(columns={dim: "bin"})

        table["bin"] = table["bin"].astype(str)
        table["dim"] = dim
        tables.append(table)

    if tables:
        combined = pd.concat(tables, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["symbol", "bin", "dim"])

    combined["timeframe"] = cfg.timeframe or ""
    combined["period_start"] = period_start
    combined["period_end"] = period_end
    combined["horizon"] = horizon
    return combined


def _score_column(measure: str, table: pd.DataFrame) -> str | None:
    preferred = ["p_hat", "lift"] if measure == "direction" else ["ret_mean", "lift"]
    for col in preferred:
        if col in table.columns:
            return col
    return None


def _rules_from_profiles_pandas(
    profiles_df: pd.DataFrame,
    cfg: spec.NormalisedSeasonalitySpec,
) -> profiles.SeasonalityRules:
    if profiles_df.empty:
        return profiles.SeasonalityRules(metadata={"thresholds": {}, "counts": {}})
    requested_dims = list(cfg.signal.dims) if cfg.signal.dims is not None else []
    if not requested_dims:
        requested_dims = list(profiles_df["dim"].dropna().unique())

    normalised_threshold = profiles._normalise_threshold(  # type: ignore[attr-defined]
        cfg.signal.threshold, cfg.profile.measure
    )
    active: Dict[str, set[Any]] = {}
    thresholds_meta: Dict[str, float | None] = {}
    counts_meta: Dict[str, int] = {}

    insufficient_mask = (
        ~profiles_df["insufficient"].fillna(False)
        if "insufficient" in profiles_df.columns
        else pd.Series(True, index=profiles_df.index)
    )
    for dim in requested_dims:
        table = profiles_df[(profiles_df["dim"] == dim) & insufficient_mask]
        if table.empty:
            active[dim] = set()
            thresholds_meta[dim] = normalised_threshold
            counts_meta[dim] = 0
            continue
        score_col = _score_column(cfg.profile.measure, table)
        if score_col is None or score_col not in table.columns:
            active[dim] = set()
            thresholds_meta[dim] = normalised_threshold
            counts_meta[dim] = 0
            continue
        table = table[table[score_col].notna()]
        if table.empty:
            active[dim] = set()
            thresholds_meta[dim] = normalised_threshold
            counts_meta[dim] = 0
            continue
        if cfg.signal.method == "threshold":
            thr = normalised_threshold if normalised_threshold is not None else float("-inf")
            filtered = table[table[score_col] >= thr]
            cutoff = thr
        else:
            k = max(int(cfg.signal.topk), 0)
            if k == 0:
                active[dim] = set()
                thresholds_meta[dim] = normalised_threshold
                counts_meta[dim] = 0
                continue
            filtered = table.sort_values(score_col, ascending=False).head(k)
            cutoff = filtered[score_col].min() if not filtered.empty else None
        bins = set(filtered["bin"].dropna().tolist())
        active[dim] = bins
        thresholds_meta[dim] = cutoff if cutoff is not None else normalised_threshold
        counts_meta[dim] = len(bins)

    metadata: Dict[str, Any] = {
        "thresholds": thresholds_meta,
        "counts": counts_meta,
        "method": cfg.signal.method,
        "measure": cfg.profile.measure,
    }
    return profiles.SeasonalityRules(active_bins=active, combine=cfg.signal.combine, metadata=metadata)


def _default_rules_metadata(rules: profiles.SeasonalityRules) -> Dict[str, Any]:
    meta = dict(rules.metadata)
    meta.setdefault("thresholds", {})
    meta.setdefault("counts", {dim: len(bins) for dim, bins in rules.active_bins.items()})
    return meta


def _attach_mi_labels(dataset: pl.DataFrame, enabled: bool) -> pl.DataFrame:
    """Attach market-intelligence label columns keyed by symbol/timestamp."""

    if not enabled or dataset.is_empty() or "timestamp" not in dataset.columns or "symbol" not in dataset.columns:
        return dataset

    dataset = _ensure_utc_timestamp(dataset, "timestamp")

    labels_frames: list[pl.DataFrame] = []
    for symbol in dataset.get_column("symbol").drop_nulls().unique().to_list():
        symbol_df = dataset.filter(pl.col("symbol") == symbol).sort("timestamp")
        if symbol_df.is_empty():
            continue
        symbol_pd = symbol_df.select(["timestamp", "open", "high", "low", "close", "volume"]).to_pandas()
        symbol_pd = symbol_pd.rename(columns={"timestamp": "ts"})
        features = mi_compute_features(symbol_pd)
        labels = mi_label_regimes(features).reset_index().rename(columns={"ts": "timestamp"})
        labels["symbol"] = symbol
        labels_pl = pl.from_pandas(labels)
        labels_pl = _ensure_utc_timestamp(labels_pl, "timestamp")
        labels_frames.append(labels_pl)

    if not labels_frames:
        return dataset

    all_labels = pl.concat(labels_frames, how="vertical", rechunk=True)
    all_labels = _ensure_utc_timestamp(all_labels, "timestamp")
    join_cols = ["symbol", "timestamp"]
    label_cols = [col for col in all_labels.columns if col.startswith("label_")]
    if not label_cols:
        return dataset
    return dataset.join(all_labels.select(join_cols + label_cols), on=join_cols, how="left", coalesce=True)


def _compute_profiles(
    dataset: pl.DataFrame,
    cfg: spec.NormalisedSeasonalitySpec,
    fold_dir: Path | None,
) -> pl.DataFrame:
    horizon_features = compute.prepare_features(dataset, cfg.profile)
    horizon_features = _attach_mi_labels(horizon_features, bool(getattr(cfg.profile, "segment_by_mi_labels", False)))
    profiles_path = str(fold_dir) if fold_dir is not None else None
    return compute.compute_profiles(
        horizon_features,
        cfg.profile,
        timeframe=cfg.timeframe,
        period_start=None,
        period_end=None,
        artifacts_out_dir=profiles_path,
        segment_by_mi_labels=bool(getattr(cfg.profile, "segment_by_mi_labels", False)),
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
    profiles_df: Any,
    cfg: spec.NormalisedSeasonalitySpec,
) -> List[Dict[str, Any]]:
    """Convert the best profiles dataframe into persistence-ready rows."""

    if profiles_df is None:
        return []
    if hasattr(profiles_df, "is_empty") and profiles_df.is_empty():
        return []
    if isinstance(profiles_df, pd.DataFrame) and profiles_df.empty:
        return []

    measure = cfg.profile.measure
    timeframe = cfg.timeframe
    start = cfg.start.date().isoformat()
    end = cfg.end.date().isoformat()
    spec_id = cfg.persistence.spec_id
    dataset_id = cfg.persistence.dataset_id

    if isinstance(profiles_df, pd.DataFrame):
        rows = profiles_df.to_dict("records")
    else:
        rows = profiles_df.to_dicts()
    records: List[Dict[str, Any]] = []
    for row in rows:
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


def _build_signals_pandas(
    rows: List[Dict[str, Any]],
    rules: profiles.SeasonalityRules,
) -> List[int]:
    df = _rows_to_pandas(rows)
    if df.empty:
        return []
    df = _add_time_bins_pandas(df)
    masks: list[pd.Series] = []
    for dim, bins in rules.active_bins.items():
        column = DIMENSION_TO_COLUMN.get(dim)
        if column is None or column not in df.columns or not bins:
            continue
        bin_values = [str(val) for val in bins]
        masks.append(df[column].astype(str).isin(bin_values))
    if not masks:
        long_mask = pd.Series(False, index=df.index)
    elif rules.combine == "and":
        long_mask = masks[0].copy()
        for mask in masks[1:]:
            long_mask &= mask
    elif rules.combine == "or":
        long_mask = masks[0].copy()
        for mask in masks[1:]:
            long_mask |= mask
    else:
        threshold = int(rules.metadata.get("sum_threshold", 1)) if rules.metadata else 1
        mask_sum = sum(mask.astype(int) for mask in masks)
        long_mask = mask_sum >= threshold
    return [1 if bool(v) else 0 for v in long_mask.tolist()]


def _atr_settings(tp_sl) -> tuple[float, float]:
    atr_mult = float(tp_sl.stop_loss) if tp_sl.stop_loss is not None else 1.0
    r_mult = float(tp_sl.take_profit) if tp_sl.take_profit is not None else 1.0
    return atr_mult, r_mult


def run(spec_model: SeasonalitySpec) -> Dict[str, Any]:
    """Execute a seasonality workflow and return aggregated metrics."""

    use_polars = pl is not None
    cfg = spec.normalise(spec_model)
    logger.info(
        "Seasonality run started | symbols=%s timeframe=%s source_path=%s mysql=%s use_polars=%s persistence=%s artifacts=%s",
        list(spec_model.data.symbols or []),
        cfg.timeframe,
        spec_model.data.dataset_path,
        bool(spec_model.data.mysql),
        use_polars,
        bool(cfg.persistence.enabled),
        bool(cfg.artifacts.out_dir),
    )
    df_source = load_ohlcv(spec_model.data)
    logger.info("Seasonality data loaded | rows=%s", len(df_source))
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
        logger.info("Seasonality artifacts root prepared | out_dir=%s", artifact_root)

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
        logger.info("Seasonality persistence run created | run_id=%s", run_id)

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
    logger.info("Seasonality split plan | folds=%s", len(folds))

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

                if use_polars:
                    train_df = _rows_to_polars(train_rows)
                    if train_df.is_empty():
                        continue
                else:
                    train_df = _rows_to_pandas(train_rows)
                    if train_df.empty:
                        continue

                fold_dir = artifact_root / f"fold_{idx}" if artifact_root is not None else None
                if fold_dir is not None:
                    fold_dir.mkdir(parents=True, exist_ok=True)

                if use_polars:
                    profiles_df = _compute_profiles(train_df, cfg, fold_dir)
                    if profiles_df.is_empty():
                        continue
                    rules = _rules_from_profiles(profiles_df, cfg)
                    signals = _build_signals(test_rows, rules)
                else:
                    train_features = _prepare_features_pandas(train_df, cfg.profile)
                    profiles_df = _compute_profiles_pandas(train_features, cfg, fold_dir)
                    if profiles_df.empty:
                        continue
                    rules = _rules_from_profiles_pandas(profiles_df, cfg)
                    signals = _build_signals_pandas(test_rows, rules)
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
                logger.info(
                    "Seasonality fold computed | fold=%s trades=%s sharpe=%s",
                    idx,
                    metrics.get("n_trades", 0),
                    metrics.get("sharpe"),
                )

                meets_min_trades = metrics["n_trades"] >= cfg.validation.min_trades
                is_better = (
                    meets_min_trades
                    and (
                        best_result is None
                        or metrics.get("sharpe", 0.0)
                        > best_result.metrics.get("sharpe", 0.0)
                    )
                )

                profiles_path = None
                if use_polars and fold_dir is not None:
                    profiles_path = fold_dir / "seasonality_profiles.parquet"

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
                    best_profiles_df = (
                        profiles_df.clone()
                        if use_polars
                        else profiles_df.copy(deep=True)
                    )

            if best_result is None:
                logger.info("Seasonality run completed without eligible best fold")
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
                    logger.info("Seasonality profiles prepared for persistence | rows=%s", len(profiles_records))
    except Exception:
        run_status = "failed"
        logger.exception("Seasonality run failed")
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
            logger.info(
                "Seasonality persistence finalized | run_id=%s status=%s profiles_rows=%s",
                run_id,
                run_status,
                len(profiles_records),
            )

    if run_id is not None:
        result_payload = dict(result_payload)
        result_payload["run_id"] = run_id

    logger.info("Seasonality run completed | run_id=%s folds=%s", run_id, len(fold_summaries))
    return result_payload
