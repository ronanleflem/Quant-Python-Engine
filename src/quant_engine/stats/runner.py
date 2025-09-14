"""Execution utilities for :mod:`quant_engine.stats`."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..api.schemas import StatsSpec
from ..core.dataset import load_dataset
from ..core.spec import DataSpec
from ..core.features import atr
from . import conditions as cond_mod
from . import events as event_mod
from . import targets as tgt_mod


def _build_data_frame(dataset: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(dataset)
    if df.empty:
        return df
    df.rename(columns={"timestamp": "ts", "session": "session_id"}, inplace=True)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def run_stats(spec: StatsSpec) -> pd.DataFrame:
    """Run statistics computation according to ``spec``.

    Returns a long-form :class:`pandas.DataFrame` with at least the columns
    ``ts``, ``symbol``, ``event``, ``event_on``, ``condition_name``,
    ``condition_value``, ``target`` and ``outcome_value``.
    """

    data_spec = DataSpec(
        path=spec.data.dataset_path,
        symbols=spec.data.symbols,
        start=date.fromisoformat(spec.data.start),
        end=date.fromisoformat(spec.data.end),
    )
    dataset = load_dataset(data_spec)
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


__all__ = ["run_stats"]

