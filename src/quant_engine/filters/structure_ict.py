import numpy as np
import pandas as pd
from typing import Optional, Literal

try:
    from quant_engine.levels import repo as lvl_repo
    from quant_engine.levels import helpers as lvl_helpers
except Exception:
    lvl_repo = None
    lvl_helpers = None


def _recent_extreme(series: pd.Series, window: int, side: str) -> pd.Series:
    """
    Retourne la série du plus haut/bas récent sur 'window' barres (rolling).
    side='high' -> rolling_max ; 'low' -> rolling_min
    Décale d'1 barre pour éviter le lookahead.
    """
    if side == "high":
        return series.rolling(window=window, min_periods=1).max().shift(1)
    else:
        return series.rolling(window=window, min_periods=1).min().shift(1)


def liquidity_sweep_filter(
    df: pd.DataFrame,
    side: Literal["high", "low"] = "high",
    lookback: int = 50,
    require_close_back_in: bool = True,
    tolerance_ticks: int = 0,
    price_increment: Optional[float] = None,
) -> pd.Series:
    """
    True si la barre actuelle 'prend la liquidité' d’un extrême récent :
      - side='high' : high[t] dépasse le plus haut des 'lookback' barres précédentes (décalées d’1),
      - side='low'  : low[t]  casse le plus bas des 'lookback' barres précédentes.
    Options :
      - require_close_back_in=True : close revient dans le range (mèche de sweep) => filtre plus strict.
      - tolerance_ticks : marge de tolérance (ticks * price_increment si fourni).
    Sans dépendre de la DB (pure OHLC). Idempotent, sans lookahead.
    """
    hi, lo, cl = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tol = (tolerance_ticks or 0) * (price_increment or 1.0)

    if side == "high":
        ref = _recent_extreme(hi, lookback, "high")
        took = (hi >= (ref + tol))
        back_in = cl <= ref if require_close_back_in else pd.Series(True, index=df.index)
        out = took & back_in
    else:
        ref = _recent_extreme(lo, lookback, "low")
        took = (lo <= (ref - tol))
        back_in = cl >= ref if require_close_back_in else pd.Series(True, index=df.index)
        out = took & back_in

    return out.fillna(False)


def _fractals(df: pd.DataFrame, left: int = 2, right: int = 2):
    """
    Fractals n-bar (fallback local si pas de levels) :
    - Swing High si high[t] > max(high[t-left...t-1]) et > max(high[t+1...t+right])
    - Swing Low idem sur low
    Retourne deux Series bool (is_swing_high, is_swing_low).
    """
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for t in range(left, n - right):
        if hi[t] > hi[t-left:t].max() and hi[t] > hi[t+1:t+1+right].max():
            sh[t] = True
        if lo[t] < lo[t-left:t].min() and lo[t] < lo[t+1:t+1+right].min():
            sl[t] = True
    return pd.Series(sh, index=df.index), pd.Series(sl, index=df.index)


def bos_filter(
    df: pd.DataFrame,
    direction: Literal["up", "down"] = "up",
    left: int = 2,
    right: int = 2,
    use_levels: bool = True,
    symbol: Optional[str] = None,
) -> pd.Series:
    """
    True si cassure de structure dans le sens demandé :
      - direction='up'   : close casse le dernier swing high (résistance)
      - direction='down' : close casse le dernier swing low (support)
    Stratégie :
      - si use_levels et SWING_H/L dispo dans marketdata.levels -> utiliser helpers/join_levels pour dernier swing.
      - sinon fallback fractals local (_fractals).
    """
    cl = df["close"].astype(float)

    if use_levels and lvl_repo is not None and lvl_helpers is not None and symbol is not None:
        try:
            lv = lvl_repo.select_levels(
                engine=None, table_fqn=None,
                symbol=symbol, level_types=["SWING_H","SWING_L"],
                active_only=False,
                start=df.index.min().isoformat(),
                end=df.index.max().isoformat(),
                limit=200000
            )
            if isinstance(lv, pd.DataFrame) and not lv.empty:
                swings = lv[lv["level_type"].isin(["SWING_H","SWING_L"])].copy()
                swings = swings.sort_values("anchor_ts")
                # asof-join pour récupérer le dernier swing connu
                base = df[["close"]].copy()
                base["ts"] = base.index
                j = lvl_helpers.join_levels(
                    base,
                    swings.rename(columns={"price": "level_price"}),
                    how="asof",
                    on="ts",
                    by=None,
                )
                j.index = df.index
                ref = j["level_price"].reindex(df.index)
                if direction == "up":
                    # dernier swing high seulement
                    mask = j["level_type"].fillna("") == "SWING_H"
                    ref_up = ref.where(mask)  # NaN si dernier level était swing low
                    ref_up = ref_up.ffill()   # propage le dernier swing high observé
                    return (cl > ref_up).fillna(False)
                else:
                    mask = j["level_type"].fillna("") == "SWING_L"
                    ref_dn = ref.where(mask)
                    ref_dn = ref_dn.ffill()
                    return (cl < ref_dn).fillna(False)
        except Exception:
            pass

    # Fallback local avec fractals
    is_sh, is_sl = _fractals(df, left=left, right=right)
    # construire la référence "dernier swing" au fil de l’eau
    ref_up = df["high"].where(is_sh).ffill()
    ref_dn = df["low"].where(is_sl).ffill()

    if direction == "up":
        return (cl > ref_up).fillna(False)
    else:
        return (cl < ref_dn).fillna(False)


def mss_filter(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    window: int = 50,
    use_levels: bool = True,
    symbol: Optional[str] = None,
) -> pd.Series:
    """
    Market Structure Shift : détection d’un 'flip' de structure :
      - on observe une cassure haussière (BOS up), suivie dans 'window' barres d’une cassure baissière (BOS down)
        OU l’inverse (down -> up). Retourne True sur la barre qui réalise le 2e break.
    Implémentation MVP :
      - construire deux Series bool : bos_up, bos_down (via bos_filter)
      - MSS = (bos_up & bos_down.rolling(window).max().shift(1)) OR (bos_down & bos_up.rolling(window).max().shift(1))
    """
    bos_up = bos_filter(df, direction="up", left=left, right=right, use_levels=use_levels, symbol=symbol)
    bos_dn = bos_filter(df, direction="down", left=left, right=right, use_levels=use_levels, symbol=symbol)

    prev_up = bos_up.rolling(window, min_periods=1).max().shift(1).fillna(0).astype(bool)
    prev_dn = bos_dn.rolling(window, min_periods=1).max().shift(1).fillna(0).astype(bool)

    mss = (bos_up & prev_dn) | (bos_dn & prev_up)
    return mss.fillna(False)
