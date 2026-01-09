"""Contradictory signals filter."""
from __future__ import annotations

import numpy as np
import pandas as pd


def contradictory_signals_filter(
    df: pd.DataFrame,
    rsi_window: int = 14,
    stoch_window: int = 14,
    williams_window: int = 14,
    z_window: int = 20,
    ema_fast: int = 20,
    ema_slow: int = 50,
    max_conflicts: int = 0,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """Return True when conflicting indicator signals are below max_conflicts."""
    for col in (price_col, high_col, low_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column: {col}")

    close = df[price_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    # RSI
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / float(rsi_window), adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / float(rsi_window), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_bull = rsi < 30
    rsi_bear = rsi > 70

    # MACD
    ema_fast_series = close.ewm(span=int(ema_fast), adjust=False).mean()
    ema_slow_series = close.ewm(span=int(ema_slow), adjust=False).mean()
    macd = ema_fast_series - ema_slow_series
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_bull = macd > macd_signal
    macd_bear = macd < macd_signal

    # Stochastic %K
    low_min = low.rolling(stoch_window, min_periods=stoch_window).min()
    high_max = high.rolling(stoch_window, min_periods=stoch_window).max()
    stoch = (close - low_min) / (high_max - low_min).replace(0.0, np.nan) * 100.0
    stoch_bull = stoch < 20
    stoch_bear = stoch > 80

    # Williams %R
    williams = (high_max - close) / (high_max - low_min).replace(0.0, np.nan) * -100.0
    will_bull = williams < -80
    will_bear = williams > -20

    # Z-score
    z_mean = close.rolling(z_window, min_periods=z_window).mean()
    z_std = close.rolling(z_window, min_periods=z_window).std(ddof=0)
    z = (close - z_mean) / z_std.replace(0.0, np.nan)
    z_bull = z < -2.0
    z_bear = z > 2.0

    # EMA trend
    ema_bull = ema_fast_series > ema_slow_series
    ema_bear = ema_fast_series < ema_slow_series

    bullish = rsi_bull + macd_bull + stoch_bull + will_bull + z_bull + ema_bull
    bearish = rsi_bear + macd_bear + stoch_bear + will_bear + z_bear + ema_bear
    conflicts = np.minimum(bullish.astype(int), bearish.astype(int))
    return (conflicts <= int(max_conflicts)).fillna(False)


__all__ = ["contradictory_signals_filter"]
