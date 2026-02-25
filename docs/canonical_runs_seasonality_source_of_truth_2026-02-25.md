# Canonical Seasonality Source Of Truth (2026-02-25)

This document is the Python runtime reference for canonical `POST /runs` requests with `spec_type = "seasonality"`.

## Scope

- Endpoint: `/runs`
- Worker flow: `QUEUED -> RUNNING -> SUCCEEDED | FAILED | CANCELED`
- Runtime path: canonical request -> seasonality spec -> `seasonality_runner.run`

## Symbol resolution

- Canonical input accepts both `data.symbol` and `data.symbols`.
- Runtime priority is strict: `data.symbols` (non-empty) > `data.symbol`.
- Validation rule: at least one of `data.symbol` or `data.symbols` must be provided.

## Supported matrix (Seasonality canonical)

- required:
  - `spec_type = seasonality`
  - `catalog_version`
  - `data.timeframe`
  - `seasonality.profile`
  - `seasonality.signal`
  - `data.symbol` or `data.symbols`
- supported:
  - `seasonality.compute`
  - `data.path` / `data.dataset_path`
  - `data.mysql`
  - `output`
  - `persistence`
- accepted but currently not wired in runtime behavior:
  - `data.asset_class`
  - `data.currency`
  - `data.window`
  - `seasonality.execution`
  - `seasonality.risk`
  - `seasonality.tp_sl`

## Runtime requirements

- Runtime data source must be available:
  - `data.path` or `data.dataset_path`, or
  - `data.mysql`
- If no valid data source is available, run fails with `execution_error`.

## Minimal payloads

Single symbol:

```json
{
  "spec_type": "seasonality",
  "catalog_version": "2026-02-02",
  "data": {
    "symbol": "SPY",
    "timeframe": "1d",
    "path": "tests/data/ohlcv_ts.csv"
  },
  "seasonality": {
    "profile": {"id": "by_hour"},
    "signal": {"method": "threshold"}
  }
}
```

Multi-symbol:

```json
{
  "spec_type": "seasonality",
  "catalog_version": "2026-02-02",
  "data": {
    "symbols": ["SPY", "QQQ"],
    "timeframe": "1d",
    "path": "tests/data/ohlcv_ts.csv"
  },
  "seasonality": {
    "profile": {"id": "by_hour"},
    "signal": {"method": "threshold"}
  }
}
```
