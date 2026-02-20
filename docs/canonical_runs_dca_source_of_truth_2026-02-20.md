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

- `symbol`, `timeframe`, `start_date`, `end_date`: required and mapped
- `dataset_path` or `path`: optional; mapped to internal CSV source
- `mysql`: optional; mapped to internal source config

### `strategy`

- `strategy.type`: required (`dca_equity` / `dca_etf`)
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

## Frontend guidance

- Prefer sending explicit `strategy.params.grid` instead of `strategy.grid` presets.
- If UI exposes `grid_conservative` / `grid_aggressive`, convert them to explicit `params.grid` client-side.
- Prefer explicit `tp_sl` object over preset strings.

