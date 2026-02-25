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
  - `stats.event`, `stats.condition`, `stats.target`
  - `data.symbol` or `data.symbols`
- supported:
  - `data.path` / `data.dataset_path`
  - `data.mysql`
  - `stats.validation`
  - `output`
  - `persistence`
- accepted but currently not wired in runtime behavior:
  - `data.lookback`
  - `data.stats_pack`
  - `data.session`
  - `data.include_weekends`
  - `data.asset_class`
  - `data.currency`

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
