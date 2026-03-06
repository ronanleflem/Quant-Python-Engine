# Canonical Market Stats Source Of Truth (2026-02-25)

This document is the Python runtime reference for canonical `POST /runs` requests with `spec_type = "market_stats"`.

## Scope

- Endpoint: `/runs`
- Worker flow: `QUEUED -> RUNNING -> SUCCEEDED | FAILED | CANCELED`
- Runtime path: canonical request -> stats spec -> `stats_runner.run_stats`

## Symbol resolution

- Canonical input accepts both `data.symbol` and `data.symbols`.
- Runtime priority is strict: `data.symbols` (non-empty) > `data.symbol`.
- Validation rule: at least one of `data.symbol` or `data.symbols` must be provided.

## Supported matrix (Market Stats canonical)

- required:
  - `spec_type = market_stats`
  - `catalog_version`
  - `data.timeframe`
  - `data.symbol` or `data.symbols`
- request modes:
  - single triplet:
    - `stats.event`, `stats.condition`, `stats.target`
  - stats pack:
    - `data.stats_pack`
    - optional `stats.condition`
    - optional `stats.validation`
- supported:
  - `data.path` / `data.dataset_path`
  - `data.mysql`
  - `data.lookback`
  - `data.stats_pack`
  - `data.asset_class`
  - `data.currency`
  - `stats.validation`
  - `output`
  - `persistence`
- accepted but currently not wired in runtime behavior:
  - `data.session`
  - `data.include_weekends`

## Stats pack catalog

- `candle_structure`
  - events: `bullish_candle`, `bearish_candle`, `bullish_engulfing`, `bearish_engulfing`, `bullish_streak(k=3)`, `bearish_streak(k=3)`
  - targets: `next_bullish`, `next_bearish`, `body_ratio`, `upper_wick_ratio`, `lower_wick_ratio`
- `volatility_shocks`
  - events: `shock_atr(mult=2.0, window=14)`, `k_consecutive(k=2, direction=up|down)`
  - targets: `up_next_bar`, `continuation_n(n=3, direction=up|down)`, `time_to_reversal(max_horizon=5)`, `candle_zscore(window=20)`
- `gaps_breakouts`
  - events: `gap_up`, `gap_down`, `breakout_hhll(lookback=20, type=high|low)`
  - targets: `up_next_bar`, `breakout_high_first`, `breakout_low_first`, `retracement_probability(direction=up|down)`
- `all_basic`
  - deterministic union of the three packs above

## Persistence shape

- `market_stats` persists both legacy and enriched metrics:
  - frequentist: `p_hat`, `ci_low`, `ci_high`, `lift`, `lift_freq`
  - Bayesian: `p_mean`, `p_map`, `hdi_low`, `hdi_high`, `lift_bayes`
  - multiple-testing / exploitation flags: `p_value`, `q_value`, `significant`, `insufficient`

## Runtime requirements

- Runtime data source must be available:
  - `data.path` or `data.dataset_path`, or
  - `data.mysql`
- If no valid data source is available, run fails with `execution_error`.

## Minimal payloads

Single symbol:

```json
{
  "spec_type": "market_stats",
  "catalog_version": "2026-02-02",
  "data": {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "path": "tests/data/ohlcv_ts.csv"
  },
  "stats": {
    "event": {"id": "always_true", "params": {}},
    "condition": {"id": "day_of_week", "params": {}},
    "target": {"id": "up_next_bar", "params": {}}
  }
}
```

Multi-symbol:

```json
{
  "spec_type": "market_stats",
  "catalog_version": "2026-02-02",
  "data": {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "timeframe": "1h",
    "path": "tests/data/ohlcv_ts.csv"
  },
  "stats": {
    "event": {"id": "always_true", "params": {}},
    "condition": {"id": "day_of_week", "params": {}},
    "target": {"id": "up_next_bar", "params": {}}
  }
}
```

Stats pack:

```json
{
  "spec_type": "market_stats",
  "catalog_version": "2026-02-02",
  "data": {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "timeframe": "1h",
    "path": "tests/data/ohlcv_ts.csv",
    "stats_pack": "all_basic"
  },
  "stats": {
    "condition": {"id": "day_of_week", "params": {}},
    "validation": {"train_months": 6, "test_months": 2, "folds": 2, "embargo_days": 0}
  }
}
```
