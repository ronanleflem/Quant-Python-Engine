import numpy as np
import pandas as pd
from typing import Optional, Literal

try:
    from quant_engine.levels import repo as lvl_repo
    from quant_engine.levels import helpers as lvl_helpers
except Exception:
    lvl_repo = None
    lvl_helpers = None


def _zscore(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window, min_periods=max(3, window//3)).mean()
    s = x.rolling(window, min_periods=max(3, window//3)).std(ddof=0)
    return (x - m) / s.replace(0, np.nan)


def volume_surge_filter(
    df: pd.DataFrame,
    window: int = 20,
    mode: Literal["z", "ratio"] = "z",
    z_thresh: float = 1.5,
    ratio_thresh: float = 1.5,
    volume_col: str = "volume",
) -> pd.Series:
    """
    True si pic de volume.
    - mode="z": z-score(volume) >= z_thresh
    - mode="ratio": volume / moyenne_rolling >= ratio_thresh
    Fallback : si pas de colonne volume -> False partout.
    """
    if volume_col not in df.columns:
        return pd.Series(False, index=df.index)
    v = df[volume_col].astype(float)
    if mode == "z":
        z = _zscore(v, window)
        out = z >= z_thresh
    else:
        ma = v.rolling(window, min_periods=max(3, window // 3)).mean()
        ratio = v / ma.replace(0, np.nan)
        out = ratio >= ratio_thresh
    return out.fillna(False)


def _compute_intraday_vwap(df: pd.DataFrame, price_col="close", volume_col="volume") -> pd.Series:
    """
    VWAP cumulatif par JOUR (UTC). Si volume absent : proxy TPO (moyenne cumulée du prix).
    """
    grp = df.index.normalize()
    if volume_col not in df.columns:
        cum_price = df.groupby(grp)[price_col].cumsum()
        counts = df.groupby(grp).cumcount() + 1
        return (cum_price / counts).reindex(df.index)
    pv = (df[price_col] * df[volume_col]).groupby(grp).cumsum()
    vv = df.groupby(grp)[volume_col].cumsum().replace(0, np.nan)
    return (pv / vv).reindex(df.index)


def vwap_side_filter(
    df: pd.DataFrame,
    anchor: Literal["day", "session"] = "day",
    side: Literal["above", "below"] = "above",
    price_col: str = "close",
    volume_col: str = "volume",
    from_levels: bool = True,
    symbol: Optional[str] = None,
) -> pd.Series:
    """
    True si le prix est du 'bon côté' du VWAP ancré:
    - anchor="day" ou "session"
    - from_levels=True : tente de lire un VWAP depuis marketdata.levels ; fallback -> calcul local
    Hypothèses : index UTC croissant; symbol optionnel pour la lecture DB.
    """
    price = df[price_col].astype(float)
    vwap_series = None

    if from_levels and lvl_repo is not None and symbol is not None:
        try:
            level_types = ["VWAP_DAY"] if anchor == "day" else ["VWAP_SESSION"]
            lv = lvl_repo.select_levels(
                engine=None, table_fqn=None,
                symbol=symbol, level_types=level_types,
                active_only=False,
                start=df.index.min().isoformat(),
                end=df.index.max().isoformat(),
                limit=100000
            )
            if isinstance(lv, pd.DataFrame) and not lv.empty and lvl_helpers is not None:
                vw = lv[lv["price"].notna()].copy().sort_values("anchor_ts")
                # asof join: propager la dernière valeur connue
                left = df[[price_col]].copy()
                left["ts"] = df.index
                joined = lvl_helpers.join_levels(
                    left,
                    vw.rename(columns={"price": "level_price"}),
                    how="asof",
                    on="ts",
                )
                if isinstance(joined, pd.DataFrame) and "level_price" in joined.columns:
                    vwap_series = joined["level_price"].reindex(df.index)
        except Exception:
            vwap_series = None

    if vwap_series is None:
        vwap_series = _compute_intraday_vwap(df, price_col=price_col, volume_col=volume_col)

    return ((price >= vwap_series) if side == "above" else (price <= vwap_series)).fillna(False)


def poc_distance_filter(
    df: pd.DataFrame,
    max_distance: float,
    unit: Literal["abs", "pct"] = "abs",
    symbol: Optional[str] = None,
    level_type: str = "POC",
) -> pd.Series:
    """
    True si la distance à un POC 'actif' le plus proche est <= max_distance.
    - Lit les POC depuis marketdata.levels si possible, sinon False (pas de fallback fiable).
    - unit='abs' : distance en prix ; unit='pct' : |close-POC|/close
    """
    if lvl_repo is None or lvl_helpers is None or symbol is None:
        return pd.Series(False, index=df.index)

    try:
        lv = lvl_repo.select_levels(
            engine=None, table_fqn=None,
            symbol=symbol, level_types=[level_type],
            active_only=True,
            start=df.index.min().isoformat(),
            end=df.index.max().isoformat(),
            limit=100000
        )
        if not isinstance(lv, pd.DataFrame) or lv.empty:
            return pd.Series(False, index=df.index)

        price_series = df["close"].astype(float)
        # Tente helper distance_to, sinon asof local
        try:
            dist = lvl_helpers.distance_to(df, lv.copy(), level_type=level_type, side="mid")
        except Exception:
            lv_ = lv.sort_values("anchor_ts")["anchor_ts"].to_frame().join(lv[["price"]])
            lv_ = lv_.dropna()
            lv_ = lv_.rename(columns={"anchor_ts": "ts", "price": "lvl"})
            lv_ = lv_.set_index(pd.to_datetime(lv_["ts"], utc=True))["lvl"].reindex(df.index, method="ffill")
            dist = (price_series - lv_).abs()

        if unit == "pct":
            dist = dist / price_series.replace(0, np.nan)

        return (dist <= max_distance).fillna(False)
    except Exception:
        return pd.Series(False, index=df.index)
