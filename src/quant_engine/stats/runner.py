from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..api.schemas import StatsSpec
from ..core.dataset import load_dataset
from ..core.spec import DataSpec
from ..core.features import atr
from ..validate import splitter
from ..io import artifacts
from ..persistence import db
from ..persistence.repo import MarketStatsRepository
from . import conditions as cond_mod
from . import events as event_mod
from . import targets as tgt_mod
from .estimators import freq_with_wilson, aggregate_binary_bayes

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
        func = getattr(event_mod, ev.name)
        col = f"ev::{ev.name}"
        df[col] = func(df, **ev.params)
        event_cols.append((ev.name, col))

    cond_cols: List[Tuple[str, str]] = []
    for cond in spec.conditions:
        func = getattr(cond_mod, cond.name)
        col = f"cond::{cond.name}"
        df[col] = func(df, **cond.params)
        cond_cols.append((cond.name, col))

    tgt_cols: List[Tuple[str, str]] = []
    for tgt in spec.targets:
        func = getattr(tgt_mod, tgt.name)
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
    data_spec = DataSpec(
        path=spec.data.dataset_path,
        symbols=spec.data.symbols,
        start=date.fromisoformat(spec.data.start),
        end=date.fromisoformat(spec.data.end),
    )
    dataset = load_dataset(data_spec)

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
    if results:
        out = pd.concat(results, ignore_index=True)[columns]
    else:
        out = pd.DataFrame(columns=columns)

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
                    "lift_freq": float(r["lift_freq"]),
                    "lift_bayes": float(r["lift_bayes"]),
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

