"""Collection of reusable pre-trade filters."""
from __future__ import annotations

from .volatility_trend import adx_filter, atr_filter, ema_slope_filter
from .volume_profile import volume_surge_filter, vwap_side_filter, poc_distance_filter
from .structure_ict import liquidity_sweep_filter, bos_filter, mss_filter
from .time_seasonality import (
    session_time_filter,
    day_of_week_filter,
    day_of_month_filter,
    month_of_year_filter,
    intraday_time_filter,
)
from .stat_prob import (
    k_consecutive_filter,
    seasonality_bin_filter,
    hurst_regime_filter,
    entropy_window_filter,
)
from .ema_structure import ema_structure_filter
from .signal_rules import rsi_entry_filter, macd_entry_filter, volume_above_average_filter
from .fractal_analysis import fractal_analysis_filter
from .mean_reversion import mean_reversion_probability_filter
from .contradictory_signals import contradictory_signals_filter
from .biais_institutional import biais_institutional_filter
from .indicator_rules import (
    atr_rising_filter,
    linear_regression_macd_cross_filter,
)
from .mtf_anomaly import mtf_anomaly_filter
from .market_regime import market_regime_filter
from .trend import trend_filter
from .stats_gate import stats_gate_filter, stats_gate_score
from .benford import benford_law_filter
from .cycles import cycles_filter
from .donchian import donchian_channels_filter
from .liquidity import liquidity_cmf_filter
from .market_manipulation import market_manipulation_filter
from .htf_poi import htf_poi_filter
from .orderflow import orderflow_delta_filter
from .macro_cot_oi import macro_cot_oi_filter
from .lt_confluence import lower_timeframe_confluence_filter
from .psychologic_news import psychologic_and_news_filter
from .ict_poi import ict_poi_filter
from .stat_arbitrage import statistical_arbitrage_filter
from .psychologic import psychologic_ulcer_filter
from .stationarity import stationarity_filter
from .volatility import volatility_filter
from .risk_mgmt import (
    daily_loss_cap_filter,
    daily_trades_cap_filter,
    cooldown_bars_filter,
    atr_risk_gate_filter,
    equity_dd_lockout_filter,
)

__all__ = [
    "adx_filter",
    "atr_filter",
    "ema_slope_filter",
    "volume_surge_filter",
    "vwap_side_filter",
    "poc_distance_filter",
    "liquidity_sweep_filter",
    "bos_filter",
    "mss_filter",
    "session_time_filter",
    "day_of_week_filter",
    "day_of_month_filter",
    "month_of_year_filter",
    "intraday_time_filter",
    "k_consecutive_filter",
    "seasonality_bin_filter",
    "hurst_regime_filter",
    "entropy_window_filter",
    "filters_registry",
    "list_filter_types",
    "daily_loss_cap_filter",
    "daily_trades_cap_filter",
    "cooldown_bars_filter",
    "atr_risk_gate_filter",
    "equity_dd_lockout_filter",
    "benford_law_filter",
    "cycles_filter",
    "donchian_channels_filter",
    "liquidity_cmf_filter",
    "statistical_arbitrage_filter",
    "psychologic_ulcer_filter",
    "stationarity_filter",
    "volatility_filter",
    "ema_structure_filter",
    "rsi_entry_filter",
    "macd_entry_filter",
    "volume_above_average_filter",
    "fractal_analysis_filter",
    "mean_reversion_probability_filter",
    "contradictory_signals_filter",
    "biais_institutional_filter",
    "atr_rising_filter",
    "linear_regression_macd_cross_filter",
    "market_regime_filter",
    "trend_filter",
    "stats_gate_filter",
    "stats_gate_score",
    "mtf_anomaly_filter",
    "market_manipulation_filter",
    "htf_poi_filter",
    "orderflow_delta_filter",
    "macro_cot_oi_filter",
    "lower_timeframe_confluence_filter",
    "psychologic_and_news_filter",
    "ict_poi_filter",
]

filters_registry = {
    "adx": adx_filter,
    "atr": atr_filter,
    "ema_slope": ema_slope_filter,
    "volume_surge": volume_surge_filter,
    "vwap_side": vwap_side_filter,
    "poc_distance": poc_distance_filter,
    "liquidity_sweep": liquidity_sweep_filter,
    "bos": bos_filter,
    "mss": mss_filter,
    "session_time": session_time_filter,
    "day_of_week": day_of_week_filter,
    "day_of_month": day_of_month_filter,
    "month_of_year": month_of_year_filter,
    "intraday_time": intraday_time_filter,
    "k_consecutive": k_consecutive_filter,
    "seasonality_bin": seasonality_bin_filter,
    "hurst_regime": hurst_regime_filter,
    "entropy_window": entropy_window_filter,
    "daily_loss_cap": daily_loss_cap_filter,
    "daily_trades_cap": daily_trades_cap_filter,
    "cooldown_bars": cooldown_bars_filter,
    "atr_risk_gate": atr_risk_gate_filter,
    "equity_dd_lockout": equity_dd_lockout_filter,
    "benford_law": benford_law_filter,
    "cycles": cycles_filter,
    "donchian_channels": donchian_channels_filter,
    "liquidity_cmf": liquidity_cmf_filter,
    "market_manipulation": market_manipulation_filter,
    "htf_poi": htf_poi_filter,
    "orderflow_delta": orderflow_delta_filter,
    "macro_cot_oi": macro_cot_oi_filter,
    "lower_timeframe_confluence": lower_timeframe_confluence_filter,
    "psychologic_and_news": psychologic_and_news_filter,
    "ict_poi": ict_poi_filter,
    "statistical_arbitrage": statistical_arbitrage_filter,
    "psychologic_ulcer": psychologic_ulcer_filter,
    "stationarity": stationarity_filter,
    "volatility": volatility_filter,
    "ema_structure": ema_structure_filter,
    "rsi_entry": rsi_entry_filter,
    "macd_entry": macd_entry_filter,
    "volume_above_average": volume_above_average_filter,
    "fractal_analysis": fractal_analysis_filter,
    "mean_reversion": mean_reversion_probability_filter,
    "contradictory_signals": contradictory_signals_filter,
    "biais_institutional": biais_institutional_filter,
    "atr_rising": atr_rising_filter,
    "linear_regression_macd_cross": linear_regression_macd_cross_filter,
    "market_regime": market_regime_filter,
    "trend": trend_filter,
    "stats_gate": stats_gate_filter,
    "mtf_anomaly": mtf_anomaly_filter,
}


def list_filter_types() -> list[str]:
    """Return the list of available filter identifiers."""

    return sorted(filters_registry)
