# Canonical DCA Source Of Truth (2026-02-20)

This document is the Python runtime reference for canonical `POST /runs` requests with `spec_type = "dca"`.

## Scope

- Endpoint: `/runs`
- Worker flow: `QUEUED -> RUNNING -> SUCCEEDED | FAILED | CANCELED`
- Runtime path: canonical request -> internal strategy spec -> `run_backtest_with_payload`

## Contract vs runtime

Validation is strict (`extra="forbid"` in canonical input models).  
A payload can pass validation and still fail at runtime if a field is accepted but not wired.

Runtime not-wired errors are returned as:

```json
{
  "error": {
    "code": "not_implemented_feature",
    "message": "Feature not implemented for canonical dca run",
    "details": [
      {"field": "<payload.path>", "reason": "accepted_but_not_wired"}
    ]
  }
}
```

## Supported matrix (DCA canonical)

### Top-level

- `spec_type`: must be `dca`
- `catalog_version`: required
- `data`: required
- `strategy`: required
- `filters`: supported (mapped to internal `filters`, `filter_rules`, `filter_rules_config`)
- `performance.initial_capital`: supported
- `performance.stress_tests`: accepted by contract, not wired in this canonical DCA runtime path

### `data`

- `symbol`, `timeframe`, `start_date`, `end_date`: mapped (symbol required only when `universe` is omitted)
- `dataset_path` or `path`: optional; mapped to internal CSV source
- `mysql`: optional; mapped to internal source config

### `strategy`

- `strategy.type`: required (`dca_equity` / `dca_etf`)
- `strategy.params.asset_class`: supported (`CRYPTO` recommended for crypto symbols)
- `strategy.params.grid` (explicit list of `{dd, weight}`): supported and preferred
- `strategy.grid` presets:
  - `grid_balanced`: supported, mapped to internal `params.grid`
  - `grid_conservative`: not wired
  - `grid_aggressive`: not wired
  - unknown preset: not wired
- `strategy.params.execution_mode`:
  - `bar_close`, `intracandle`: supported
  - other values (for example `limit`): not wired
- `strategy.params.drawdown_reference`:
  - `ATH`, `1M`, `3M`, `6M`, `1Y`: supported by strategy runtime
  - `rolling_high`: supported alias in canonical mapping, converted to `90D`

### `universe`

- `universe[]`: supported in canonical DCA input
- each item requires at least `symbol` (optional metadata: `asset_class`, `exchange`, `currency`, `broker`, ...)
- resolution priority in runtime mapping: `universe` first, then fallback `data.symbol`
- recommended mode: always send `universe[]` (including mono-symbol runs)

### `strategy.params.tp_sl`

Supported shapes:

1. Internal runtime shape (`rules` / `sl_dd`)  
2. Canonical explicit object:

```json
{
  "enabled": true,
  "mode": "rule_based",
  "tp": {"type": "percent", "value": 2.0},
  "sl": {"type": "percent", "value": 1.0},
  "break_even": {"enabled": true, "trigger_pct": 1.0}
}
```

3. Preset string format: `tp_<X>_sl_<Y>` (example: `tp_2_sl_1`)

4. Optional trailing stop (explicit object only):

```json
{
  "trailing": {"enabled": true, "type": "percent", "value": 1.0, "trigger_pct": 1.2}
}
```

Trailing rules:
- `type` must be `percent`
- `value` must be > 0
- `trigger_pct` is optional (defaults to `value`) and must be >= 0

Not wired:

- string presets outside `tp_<X>_sl_<Y>` format
- explicit object with non-percent type, missing values, or non-positive values

## Recommended payload (runtime-safe)

```json
{
  "spec_type": "dca",
  "catalog_version": "2026-02-02",
  "data": {
    "symbol": "BTCUSD",
    "timeframe": "1h",
    "start_date": "2022-12-31",
    "end_date": "2024-12-30"
  },
  "strategy": {
    "type": "dca_equity",
    "params": {
      "execution_mode": "bar_close",
      "drawdown_reference": "ATH",
      "grid": [{"dd": -5.0, "weight": 1.0}],
      "tp_sl": {
        "enabled": true,
        "mode": "rule_based",
        "tp": {"type": "percent", "value": 2.0},
        "sl": {"type": "percent", "value": 1.0}
      }
    }
  }
}
```

## Multi-symbol runtime rules

- execution scope: per symbol in `universe`
- filters/rules scope: evaluated independently per symbol on each symbol OHLC
- aggregation:
  - `result.counts` is keyed by symbol
  - backend payload aggregates generated trades/signals for the run
- missing data behavior:
  - if one symbol cannot load OHLC, the run fails (`execution_error`)

## Deprecation & migration

- `data.symbol` is still accepted for compatibility, but deprecated for canonical DCA
- replacement: `universe[]`
- deprecation warning event in worker logs: `canonical_dca_deprecation_warning`
- target removal window: `2026-06` (subject to release validation)

Migration example (mono-symbol):

Before:
```json
{
  "data": {"symbol": "BTCUSDT", "timeframe": "1h", "start_date": "2022-12-31", "end_date": "2024-12-30"}
}
```

After:
```json
{
  "data": {"timeframe": "1h", "start_date": "2022-12-31", "end_date": "2024-12-30"},
  "universe": [{"symbol": "BTCUSDT", "asset_class": "CRYPTO"}]
}
```

## Frontend guidance

- Prefer sending explicit `strategy.params.grid` instead of `strategy.grid` presets.
- If UI exposes `grid_conservative` / `grid_aggressive`, convert them to explicit `params.grid` client-side.
- Prefer explicit `tp_sl` object over preset strings.
