# Canonical Backtest Source Of Truth (2026-02-20)

This document is the Python runtime reference for canonical `POST /runs` requests with `spec_type = "backtest"`.

## Scope

- Endpoint: `/runs`
- Worker flow: `QUEUED -> RUNNING -> SUCCEEDED | FAILED | CANCELED`
- Runtime path: canonical request -> internal backtest spec -> `backtest_runner.run_backtest_from_spec`

## Supported in canonical runtime (wired)

- `data.symbol`, `data.timeframe`, `data.start_date`, `data.end_date`
- `data.dataset_path` / `data.path` (CSV source)
- `data.mysql`
- implicit source mode when no explicit source is provided (`data.path`/`data.dataset_path`/`data.mysql` absent):
  - canonical mapper enables runtime source resolution chain
  - order: Delta -> MySQL -> Java
  - MySQL env key used in auto mode: `QE_MARKETDATA_MYSQL_URL`
- `signal` with `type = ema_cross` and params `fast`, `slow`, `require_crossing`
- `filters.filters` -> internal `filters`
- `filters.rules` -> internal `filter_rules`
- `filters.rules_config` -> internal `filter_rules_config`
- `strategy.params.tp_sl` when provided as internal backtest TP/SL object (`atr_window`, `atr_k`, `r_mult`, etc.)
- `performance.initial_capital`
- `output`
- `persistence`

## Accepted but not wired

- `strategy.name`
- `performance.stress_tests`

## Runtime not-implemented behavior

When a payload contains accepted-but-not-wired fields, worker returns:

```json
{
  "error": {
    "code": "not_implemented_feature",
    "message": "Feature not implemented for canonical backtest run",
    "details": [{"field": "<payload.path>", "reason": "accepted_but_not_wired"}]
  }
}
```

## Recommended minimal payload

```json
{
  "spec_type": "backtest",
  "catalog_version": "2026-02-02",
  "data": {
    "symbol": "EURUSD",
    "timeframe": "1h",
    "start_date": "2022-12-31",
    "end_date": "2024-12-30"
  },
  "signal": {
    "type": "ema_cross",
    "fast": 12,
    "slow": 26,
    "require_crossing": true
  }
}
```

This minimal payload can run in auto source mode if your runtime environment has at least one available source
(Delta, MySQL via `QE_MARKETDATA_MYSQL_URL`, or Java OHLC endpoint).
