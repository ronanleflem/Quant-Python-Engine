"""Pure pipeline primitives for MarketIntelligenceService v1."""

from __future__ import annotations

import pandas as pd


def _utc_indexed(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.index.tz_localize("UTC") if frame.index.tz is None else frame.index.tz_convert("UTC")
    if "ts" not in frame.columns:
        raise ValueError("ohlcv must expose a DatetimeIndex or a 'ts' column")
    return pd.to_datetime(frame["ts"], utc=True)


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic correlation/regime/liquidity base features from OHLCV."""
    idx = _utc_indexed(ohlcv)
    base = ohlcv.copy()

    close = pd.to_numeric(base["close"], errors="coerce")
    volume = pd.to_numeric(base["volume"], errors="coerce")

    features = pd.DataFrame(index=idx)
    features.index.name = "ts"
    features["feat_return_1"] = close.pct_change().fillna(0.0)
    features["feat_volatility_5"] = features["feat_return_1"].rolling(5, min_periods=2).std().fillna(0.0)
    features["feat_corr_close_volume_5"] = close.rolling(5, min_periods=3).corr(volume).fillna(0.0)

    return features


def label_regimes(features: pd.DataFrame) -> pd.DataFrame:
    """Label market regime from momentum and realized volatility proxies."""
    frame = features.copy()
    trend = frame["feat_return_1"].rolling(3, min_periods=1).mean()
    high_vol = frame["feat_volatility_5"] > frame["feat_volatility_5"].rolling(8, min_periods=1).median()

    labels = pd.DataFrame(index=frame.index)
    labels.index.name = "ts"
    labels["label_regime"] = "sideways"
    labels.loc[trend > 0.002, "label_regime"] = "bull"
    labels.loc[trend < -0.002, "label_regime"] = "bear"
    labels.loc[high_vol, "label_regime"] = labels.loc[high_vol, "label_regime"] + "_high_vol"
    return labels


def liquidity_flags(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Generate deterministic liquidity flags from volume and intrabar spread."""
    idx = _utc_indexed(ohlcv)
    volume = pd.to_numeric(ohlcv["volume"], errors="coerce")
    high = pd.to_numeric(ohlcv["high"], errors="coerce")
    low = pd.to_numeric(ohlcv["low"], errors="coerce")
    close = pd.to_numeric(ohlcv["close"], errors="coerce")

    spread = ((high - low) / close.replace(0, pd.NA)).fillna(0.0)
    rolling_median_vol = volume.rolling(10, min_periods=1).median()

    flags = pd.DataFrame(index=idx)
    flags.index.name = "ts"
    flags["liq_low_volume"] = volume < (rolling_median_vol * 0.5)
    flags["liq_wide_spread"] = spread > 0.02
    flags["liq_illiquid"] = flags["liq_low_volume"] | flags["liq_wide_spread"]
    return flags


__all__ = ["compute_features", "label_regimes", "liquidity_flags"]
