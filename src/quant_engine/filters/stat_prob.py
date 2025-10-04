import numpy as np
import pandas as pd
from typing import List, Literal, Optional

try:
    # Optionnel : si un repo stats existe déjà pour lire quant.seasonality_profiles
    from quant_engine.stats import repo as stats_repo
except Exception:  # pragma: no cover - optional dependency for fallback
    stats_repo = None


def k_consecutive_filter(
    df: pd.DataFrame,
    k: int = 3,
    direction: Literal["up", "down"] = "up",
    use_body: bool = True,
) -> pd.Series:
    """Return ``True`` when the last *k* candles move in the same direction.

    ``use_body`` toggles between comparing ``close`` vs ``open`` (True) or
    ``close`` vs ``close.shift(1)`` (False).
    """

    if k <= 0:
        raise ValueError("k must be strictly positive")

    if use_body:
        delta = df["close"].astype(float) - df["open"].astype(float)
    else:
        delta = df["close"].astype(float).diff()

    sign = np.sign(delta)
    cond = (sign > 0).astype(int) if direction == "up" else (sign < 0).astype(int)
    run = cond.rolling(window=k, min_periods=k).sum()
    return (run == k).reindex(df.index).fillna(False)


def seasonality_bin_filter(
    df: pd.DataFrame,
    mode: Literal["hour", "dow", "dom", "month", "session"] = "hour",
    allowed: Optional[List[int]] = None,
    blocked: Optional[List[int]] = None,
    min_winrate: Optional[float] = None,
    symbol: Optional[str] = None,
) -> pd.Series:
    """Filter bars that fall within a seasonal bucket deemed positive."""

    idx = df.index
    if mode == "hour":
        bins = pd.Series(idx.hour, index=idx)
    elif mode == "dow":
        bins = pd.Series(idx.dayofweek, index=idx)
    elif mode == "dom":
        bins = pd.Series(idx.day, index=idx)
    elif mode == "month":
        bins = pd.Series(idx.month, index=idx)
    else:  # session buckets (Asia/London/Overlap/US)
        hours = idx.hour
        session_id = np.full(len(idx), -1)
        session_id[(hours >= 0) & (hours < 8)] = 0       # Asia 00-07
        session_id[(hours >= 8) & (hours < 13)] = 1      # London 08-12
        session_id[(hours >= 13) & (hours < 17)] = 2     # Overlap 13-16
        session_id[(hours >= 17) & (hours < 22)] = 3     # US 17-21
        bins = pd.Series(session_id, index=idx)

    if min_winrate is not None and stats_repo is not None:
        try:
            profile = stats_repo.select_seasonality_profile(
                symbol=symbol,
                dimension=mode,
            )
        except Exception:  # pragma: no cover - optional database call
            profile = None

        if isinstance(profile, pd.DataFrame) and {"bin", "winrate"}.issubset(profile.columns):
            winners = profile.loc[profile["winrate"] >= float(min_winrate), "bin"].astype(int)
            good = set(winners.tolist())
            return bins.isin(good)

    if allowed is not None:
        return bins.isin(allowed)
    if blocked is not None:
        return ~bins.isin(blocked)

    return pd.Series(True, index=idx)


def _hurst_rs(price: pd.Series, window: int) -> pd.Series:
    """Estimate the Hurst exponent via a rolling rescaled range calculation."""

    log_price = np.log(price.astype(float))
    returns = log_price.diff().dropna()
    hurst = pd.Series(np.nan, index=price.index)

    for end in range(window, len(returns) + 1):
        segment = returns.iloc[end - window : end]
        if segment.std(ddof=0) == 0:
            h_value = np.nan
        else:
            cumulative = segment.cumsum() - segment.mean() * np.arange(1, window + 1)
            r_val = cumulative.max() - cumulative.min()
            s_val = segment.std(ddof=0)
            if s_val == 0:
                h_value = np.nan
            else:
                rs = r_val / s_val
                h_value = np.log(rs) / np.log(window)
        hurst.loc[segment.index[-1]] = h_value

    return hurst.ffill()


def hurst_regime_filter(
    df: pd.DataFrame,
    window: int = 256,
    min_h: float = 0.45,
    max_h: float = 0.65,
    price_col: str = "close",
) -> pd.Series:
    """Keep bars whose Hurst exponent lies within ``[min_h, max_h]``."""

    if window <= 1:
        raise ValueError("window must be greater than 1 for Hurst calculation")

    price = df[price_col].astype(float)
    hurst = _hurst_rs(price, window)
    mask = (hurst >= float(min_h)) & (hurst <= float(max_h))
    return mask.reindex(df.index).fillna(False)


def entropy_window_filter(
    df: pd.DataFrame,
    window: int = 64,
    max_entropy: Optional[float] = None,
    min_entropy: Optional[float] = None,
    use_body: bool = False,
) -> pd.Series:
    """Filter bars according to the directional entropy over a rolling window."""

    if window <= 1:
        raise ValueError("window must be greater than 1")

    if use_body:
        direction = np.sign((df["close"] - df["open"]).astype(float))
    else:
        direction = np.sign(df["close"].astype(float).diff())

    up_prob = (direction > 0).astype(int).rolling(window, min_periods=max(8, window // 4)).mean()
    eps = 1e-12
    entropy = -(up_prob * np.log2(up_prob + eps) + (1 - up_prob) * np.log2(1 - up_prob + eps))

    cond = pd.Series(True, index=df.index)
    if max_entropy is not None:
        cond &= entropy <= float(max_entropy)
    if min_entropy is not None:
        cond &= entropy >= float(min_entropy)

    return cond.fillna(False)


__all__ = [
    "k_consecutive_filter",
    "seasonality_bin_filter",
    "hurst_regime_filter",
    "entropy_window_filter",
]
