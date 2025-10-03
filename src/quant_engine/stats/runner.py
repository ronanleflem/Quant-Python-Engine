from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..api.schemas import StatsSpec
from ..core.dataset import load_ohlcv
from ..core.features import atr
from ..validate import splitter
from ..io import artifacts
from ..persistence import db
from ..persistence.repo import MarketStatsRepository
from . import conditions as cond_mod
from . import events as event_mod
from . import targets as tgt_mod
from .estimators import (
    freq_with_wilson,
    aggregate_binary_bayes,
    p_value_binomial_onesided_normal,
    benjamini_hochberg,
)

N_MIN = 300


def _build_data_frame(dataset: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(dataset)
    if df.empty:
        return df
    df.rename(columns={"timestamp": "ts", "session": "session_id"}, inplace=True)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def _long_form(dataset: List[Dict[str, Any]], spec: StatsSpec) -> pd.DataFrame:
    df = _build_data_frame(dataset)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ts",
                "symbol",
                "event",
                "event_on",
                "condition_name",
                "condition_value",
                "target",
                "outcome_value",
            ]
        )

    df["atr"] = atr.compute(dataset, {"period": 14})

    event_cols: List[Tuple[str, str]] = []
    for ev in spec.events:
        func_name = getattr(ev, "type", None) or ev.name
        func = getattr(event_mod, func_name)
        col = f"ev::{ev.name}"
        df[col] = func(df, **ev.params)
        event_cols.append((ev.name, col))

    cond_cols: List[Tuple[str, str]] = []
    regular_cond_entries: List[Dict[str, Any]] = []
    level_cond_entries: List[Dict[str, Any]] = []
    for cond in spec.conditions:
        func_name = getattr(cond, "type", None) or cond.name
        func = getattr(cond_mod, func_name)
        col = f"cond::{cond.name}"
        cond_cols.append((cond.name, col))
        requires_levels = "_level" in func_name
        if requires_levels:
            callable_func = func(**cond.params)
            level_type = cond.params.get("level_type")
            entry = {
                "name": cond.name,
                "column": col,
                "callable": callable_func,
                "level_types": [level_type] if level_type else [],
            }
            level_cond_entries.append(entry)
        else:
            def _regular(df_local: pd.DataFrame, *, _func=func, _params=cond.params):
                return _func(df_local, **_params)

            regular_cond_entries.append({
                "name": cond.name,
                "column": col,
                "callable": _regular,
            })

    for entry in regular_cond_entries:
        df[entry["column"]] = entry["callable"](df)

    if level_cond_entries:
        aggregated_level_types = sorted({lt for entry in level_cond_entries for lt in entry["level_types"] if lt})
        for symbol, group in df.groupby("symbol"):
            if aggregated_level_types:
                levels_df = cond_mod._load_levels_for(group, aggregated_level_types)
            else:
                levels_df = pd.DataFrame()
            for entry in level_cond_entries:
                series = entry["callable"](group, levels_df=levels_df)
                df.loc[group.index, entry["column"]] = series.reindex(group.index)

    tgt_cols: List[Tuple[str, str]] = []
    for tgt in spec.targets:
        func_name = getattr(tgt, "type", None) or tgt.name
        func = getattr(tgt_mod, func_name)
        col = f"tgt::{tgt.name}"
        df[col] = func(df, **tgt.params)
        tgt_cols.append((tgt.name, col))

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        base = {"ts": row["ts"], "symbol": row["symbol"]}
        for ev_name, ev_col in event_cols:
            ev_on = bool(row[ev_col])
            for tgt_name, tgt_col in tgt_cols:
                outcome = row[tgt_col]
                if cond_cols:
                    for cond_name, cond_col in cond_cols:
                        records.append(
                            {
                                **base,
                                "event": ev_name,
                                "event_on": ev_on,
                                "condition_name": cond_name,
                                "condition_value": row[cond_col],
                                "target": tgt_name,
                                "outcome_value": outcome,
                            }
                        )
                else:
                    records.append(
                        {
                            **base,
                            "event": ev_name,
                            "event_on": ev_on,
                            "condition_name": None,
                            "condition_value": None,
                            "target": tgt_name,
                            "outcome_value": outcome,
                        }
                    )
    return pd.DataFrame.from_records(records)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "event",
        "condition_name",
        "condition_value",
        "target",
        "n",
        "successes",
        "p_hat",
        "ci_low",
        "ci_high",
        "p_mean",
        "p_map",
        "hdi_low",
        "hdi_high",
        "lift_freq",
        "lift_bayes",
        "insufficient",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df[df["event_on"]].dropna(subset=["outcome_value"])
    if df.empty:
        return pd.DataFrame(columns=columns)

    base = (
        df.groupby(["symbol", "target"], dropna=False)["outcome_value"]
        .agg(["count", "sum"])
        .rename(columns={"count": "n", "sum": "successes"})
        .reset_index()
    )
    base["baseline_freq"] = base["successes"] / base["n"]
    base["baseline_bayes"] = base.apply(
        lambda r: aggregate_binary_bayes(int(r["successes"]), int(r["n"]))[
            "p_mean"
        ],
        axis=1,
    )
    base = base[["symbol", "target", "baseline_freq", "baseline_bayes"]]

    grouped = (
        df.groupby(
            ["symbol", "event", "condition_name", "condition_value", "target"],
            dropna=False,
        )["outcome_value"]
        .agg(["count", "sum"])
        .rename(columns={"count": "n", "sum": "successes"})
        .reset_index()
    )
    if grouped.empty:
        return pd.DataFrame(columns=columns)

    grouped[["p_hat", "ci_low", "ci_high"]] = grouped.apply(
        lambda r: pd.Series(freq_with_wilson(int(r["successes"]), int(r["n"]))),
        axis=1,
    )
    grouped[["p_mean", "p_map", "hdi_low", "hdi_high"]] = grouped.apply(
        lambda r: pd.Series(
            {k: v for k, v in aggregate_binary_bayes(int(r["successes"]), int(r["n"])).items() if k in {"p_mean", "p_map", "hdi_low", "hdi_high"}}
        ),
        axis=1,
    )
    grouped = grouped.merge(base, on=["symbol", "target"], how="left")
    grouped["lift_freq"] = grouped["p_hat"] - grouped["baseline_freq"]
    grouped["lift_bayes"] = grouped["p_mean"] - grouped["baseline_bayes"]
    grouped.drop(columns=["baseline_freq", "baseline_bayes"], inplace=True)
    grouped["insufficient"] = grouped["n"] < N_MIN
    grouped.loc[
        grouped["insufficient"],
        ["ci_low", "ci_high", "hdi_low", "hdi_high"],
    ] = pd.NA
    return grouped[columns]


def run_stats(spec: StatsSpec) -> pd.DataFrame:
    df_source = load_ohlcv(spec.data)
    dataset: List[Dict[str, Any]] = []
    for row in df_source.to_dict("records"):
        rec = dict(row)
        ts = rec.pop("ts", None)
        if ts is not None:
            ts_value = pd.to_datetime(ts, utc=True)
            rec["timestamp"] = ts_value.isoformat()
        dataset.append(rec)
    dataset.sort(key=lambda r: r.get("timestamp", ""))

    splits: List[Tuple[str, List[Dict[str, Any]]]] = []
    if spec.validation is not None:
        val = spec.validation
        folds = splitter.generate_folds(
            dataset, val.train_months, val.test_months, val.folds, val.embargo_days
        )
        if not folds:
            splits.append(("test", dataset))
        else:
            for fold in folds:
                splits.append(("train", fold["train"]))
                splits.append(("test", fold["test"]))
    else:
        splits.append(("test", dataset))

    results: List[pd.DataFrame] = []
    for split_name, split_data in splits:
        df_long = _long_form(split_data, spec)
        agg = _aggregate(df_long)
        if agg.empty:
            continue
        agg["split"] = split_name
        results.append(agg)

    columns = [
        "symbol",
        "event",
        "condition_name",
        "condition_value",
        "target",
        "n",
        "successes",
        "p_hat",
        "ci_low",
        "ci_high",
        "p_mean",
        "p_map",
        "hdi_low",
        "hdi_high",
        "lift_freq",
        "lift_bayes",
        "insufficient",
        "split",
    ]
    final_columns = columns + ["p_value", "q_value", "significant"]
    if results:
        out = pd.concat(results, ignore_index=True)[columns]
    else:
        out = pd.DataFrame(columns=columns)

    if not out.empty:
        out["p_value"] = pd.NA
        out["q_value"] = pd.NA
        out["significant"] = False
        group_cols = ["symbol", "target", "split"]
        for _, idx in out.groupby(group_cols).groups.items():
            g = out.loc[idx]
            total_n = int(g["n"].sum())
            total_successes = int(g["successes"].sum())
            p0 = total_successes / total_n if total_n else 0.0
            pvals = []
            for _, row in g.iterrows():
                direction = "greater" if row["lift_freq"] >= 0 else "less"
                pval = p_value_binomial_onesided_normal(
                    int(row["successes"]), int(row["n"]), p0, direction=direction
                )
                pvals.append(pval)
            qvals = benjamini_hochberg(pvals)
            out.loc[idx, "p_value"] = pvals
            out.loc[idx, "q_value"] = qvals
            out.loc[idx, "significant"] = [q <= 0.05 for q in qvals]

    out = out.reindex(columns=final_columns)

    if spec.artifacts and spec.artifacts.out_dir:
        out_dir = Path(spec.artifacts.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts.write_stats_summary(out_dir / "stats_summary.parquet", out)
        artifacts.write_stats_details(out_dir / "stats_details.parquet", pd.DataFrame())

    if spec.persistence and getattr(spec.persistence, "enabled", False):
        rows: List[Dict[str, Any]] = []
        start = spec.data.start
        end = spec.data.end
        timeframe = spec.data.timeframe
        spec_id = getattr(spec.persistence, "spec_id", None)
        dataset_id = getattr(spec.persistence, "dataset_id", None)
        for r in out.to_dict("records"):
            cond_val = r["condition_value"]
            if pd.isna(cond_val):
                cond_val = None
            ci_low = r["ci_low"]
            if pd.isna(ci_low):
                ci_low = None
            ci_high = r["ci_high"]
            if pd.isna(ci_high):
                ci_high = None
            hdi_low = r["hdi_low"]
            if pd.isna(hdi_low):
                hdi_low = None
            hdi_high = r["hdi_high"]
            if pd.isna(hdi_high):
                hdi_high = None
            lift_freq_val = r.get("lift_freq")
            if pd.isna(lift_freq_val):
                lift_freq_val = 0.0
            lift_bayes_val = r.get("lift_bayes")
            if pd.isna(lift_bayes_val):
                lift_bayes_val = lift_freq_val

            rows.append(
                {
                    "symbol": r["symbol"],
                    "timeframe": timeframe,
                    "event": r["event"],
                    "condition_name": r["condition_name"],
                    "condition_value": cond_val,
                    "target": r["target"],
                    "split": r["split"],
                    "n": int(r["n"]),
                    "successes": int(r["successes"]),
                    "p_hat": float(r["p_hat"]),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_mean": float(r["p_mean"]),
                    "p_map": float(r["p_map"]),
                    "hdi_low": hdi_low,
                    "hdi_high": hdi_high,
                    "lift_freq": float(lift_freq_val),
                    "lift_bayes": float(lift_bayes_val),
                    "lift": float(lift_freq_val),
                    "start": start,
                    "end": end,
                    "spec_id": spec_id,
                    "dataset_id": dataset_id,
                }
            )
        if rows:
            with db.session() as conn:
                repo = MarketStatsRepository(conn)
                repo.bulk_upsert(rows)

    return out


__all__ = ["run_stats"]

