"""Parquet-backed feature store with partitioned/versioned persistence."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_engine.market_intelligence.models import FeatureMeta


class ParquetFeatureStore:
    """Persist feature dataframes in a stable parquet layout.

    Layout:
        ``<root>/<feature_name>/symbol=<symbol>/timeframe=<timeframe>/version=<feature_version>/data.parquet``
    """

    def __init__(self, root_dir: str | Path, *, timestamp_column: str = "timestamp") -> None:
        self._root_dir = Path(root_dir)
        self._timestamp_column = timestamp_column

    def _build_path(self, feature_name: str, symbol: str, timeframe: str, feature_version: str) -> Path:
        meta = FeatureMeta(
            feature_name=feature_name,
            symbol=symbol,
            timeframe=timeframe,
            feature_version=feature_version,
        )
        return (
            self._root_dir
            / meta.feature_name
            / f"symbol={meta.symbol}"
            / f"timeframe={meta.timeframe}"
            / f"version={meta.feature_version}"
            / "data.parquet"
        )

    def put(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str,
        feature_version: str,
        payload: pd.DataFrame,
        *,
        overwrite: bool = False,
    ) -> Path:
        data_path = self._build_path(feature_name, symbol, timeframe, feature_version)
        if data_path.exists() and not overwrite:
            raise KeyError(f"Feature key already exists: {data_path}")

        if "feature_version" in payload.columns and not payload["feature_version"].eq(feature_version).all():
            raise ValueError("Payload feature_version does not match requested feature_version")

        data_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = data_path.with_name(f".{data_path.name}.{uuid4().hex}.tmp")
        try:
            payload.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, data_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return data_path

    def get(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str,
        feature_version: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        data_path = self._build_path(feature_name, symbol, timeframe, feature_version)
        if not data_path.exists():
            raise KeyError(f"Feature key not found: {data_path}")

        frame = pd.read_parquet(data_path)

        if "feature_version" in frame.columns and not frame["feature_version"].eq(feature_version).all():
            raise ValueError("Stored feature_version does not match requested feature_version")

        if start is None and end is None:
            return frame

        if self._timestamp_column in frame.columns:
            timestamps = pd.to_datetime(frame[self._timestamp_column], utc=False)
            mask = pd.Series(True, index=frame.index)
            if start is not None:
                mask &= timestamps >= pd.Timestamp(start)
            if end is not None:
                mask &= timestamps <= pd.Timestamp(end)
            return frame.loc[mask].copy()

        if isinstance(frame.index, pd.DatetimeIndex):
            mask = pd.Series(True, index=frame.index)
            if start is not None:
                mask &= frame.index >= pd.Timestamp(start)
            if end is not None:
                mask &= frame.index <= pd.Timestamp(end)
            return frame.loc[mask].copy()

        raise ValueError(
            f"Cannot apply period filter: no '{self._timestamp_column}' column and index is not DatetimeIndex"
        )

    def exists(self, feature_name: str, symbol: str, timeframe: str, feature_version: str) -> bool:
        return self._build_path(feature_name, symbol, timeframe, feature_version).exists()
