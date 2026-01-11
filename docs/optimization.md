# Optimization Workflow

This document describes the optimization workflow for backtest and strategy specs.

## What is stored

Light data (always persisted per trial):
- `trial_id`, `params`
- aggregate metrics (`sharpe`, `sortino`, `returnPct`, `maxDrawdownPct`, `winratePct`, `totalReturn`)
- objective value

Heavy data (persisted only for promoted trials):
- full payload for the trial (run + trades)

Heavy payloads are written to `runs/optimize_backtest/promoted/` or
`runs/optimize_strategy/promoted/` as `trial_{id}.json`.

## Promotion policy (top-K + constraints)

Promotion is controlled by `optimization.promotion`:

```json
{
  "optimization": {
    "objective": "sharpe",
    "promotion": {
      "top_k": 3,
      "min_trades": 1,
      "max_drawdown_pct": 60,
      "min_winrate_pct": 20,
      "min_return_pct": 0,
      "min_sharpe": 0.2,
      "min_sortino": 0.2,
      "dedupe_distance": 0.15
    }
  }
}
```

- `top_k`: number of best trials to keep heavy payloads for.
- `min_trades`: minimum number of trades (wins + losses).
- `max_drawdown_pct`: max allowed drawdown percentage.
- `min_winrate_pct`: minimum win rate percentage.
- `min_return_pct`: minimum return percentage.
- `min_sharpe`: minimum Sharpe ratio.
- `min_sortino`: minimum Sortino ratio.
- `dedupe_distance`: optional diversity filter. When set, a candidate is
  rejected if it is too close to an existing promoted trial (distance is
  computed on normalized parameters using the search space bounds).

## Screening vs full run

Current flow:
1) One pass over all trials with light storage (`trials.json`).
2) Promote top-K candidates that pass constraints.
3) Persist heavy payloads only for promoted trials.

### Screening mode (early shortcut)

Screening lets you speed up optimization by using a reduced slice of data.
It is applied before filters/signals are computed.

```json
{
  "optimization": {
    "screening": {
      "enabled": true,
      "max_bars": 300,
      "max_trades": 25,
      "max_seconds": 2.0
    }
  }
}
```

When enabled:
- only the last `max_bars` are used for the trial
- stop early after `max_trades` completed trades (DCA/crypto grid)
- stop early after `max_seconds` of compute time
- storage remains light for all trials

## Remaining work

- Add early-stop (max trades, max time) and sub-window sampling.
- Add explicit dataset_id / code_version metadata in trials.
- Add alternative promotion policies (composite objective, behavior clustering).
- Add optional compressed artifacts (equity curve summary, trade stats only).
