## Title
- ANG-OPT-1 - Add optimization toggle inside DCA/Backtest forms (no separate launcher mode)

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Cross-Repo Coordination
- Cross-Repo Initiative: Optimization Launcher
- Repo Owner: angular-financial-project
- Upstream Dependencies:
  - PY-OPT-1 (`/runs` optimization contract + `/runs/capabilities`)
- Contract Version: catalog_version=2026-02-02

## Goal
- Keep a single DCA/Backtest user flow and add an optional optimization block:
  - Optimization OFF => normal `dca` / `backtest` run
  - Optimization ON => canonical `optimize_dca` / `optimize_backtest` run

## Audit findings (current state)
- Existing launcher already has DCA/Backtest forms and submit pipeline through `/api/runs`.
- Creating a brand new top-level optimization volet would duplicate strategy inputs and confuse users.
- Backend contract now expects optimization as dedicated `spec_type` (`optimize_dca`/`optimize_backtest`) with `optimization.base_spec`.

## Context / Entry points
- Modules/files:
  - `src/app/pages/strategy-launcher/strategy-launcher.page.ts`
  - `src/app/pages/strategy-launcher/strategy-launcher.page.html`
  - `src/app/models/run-request-input.model.ts`
  - `src/app/services/runs.service.ts`

## Definition of Done
- [ ] No new top-level launcher tab for optimization is added.
- [ ] DCA form includes `Enable optimization` toggle + optimization subsection.
- [ ] Backtest form includes `Enable optimization` toggle + optimization subsection.
- [ ] Optimization subsection contains:
  - objective (`metric`, `direction`)
  - budget (`max_trials`, optional `timeout_seconds`, optional `seed`)
  - search-space editor (JSON/object)
- [ ] Submit flow switches payload mode automatically:
  - OFF => `spec_type=dca|backtest`
  - ON => `spec_type=optimize_dca|optimize_backtest` + `optimization.base_spec`
- [ ] Capabilities are fetched for `optimize_dca` and `optimize_backtest` for labels/tooltips/constraints.
- [ ] Toggle OFF guarantees no optimization block is serialized.
- [ ] Toggle ON guarantees `optimization.base_spec.spec_type` matches target:
  - DCA screen => `dca`
  - Backtest screen => `backtest`

## Canonical payload contract (must-match)
1. DCA without optimization:
```json
{
  "spec_type": "dca",
  "catalog_version": "2026-02-02",
  "...": "existing dca payload"
}
```
2. DCA with optimization:
```json
{
  "spec_type": "optimize_dca",
  "catalog_version": "2026-02-02",
  "optimization": {
    "base_spec": { "spec_type": "dca", "...": "existing dca payload" },
    "search_space": {},
    "objective": { "metric": "sharpe", "direction": "max" },
    "budget": { "max_trials": 50 }
  }
}
```
3. Backtest with optimization:
```json
{
  "spec_type": "optimize_backtest",
  "catalog_version": "2026-02-02",
  "optimization": {
    "base_spec": { "spec_type": "backtest", "...": "existing backtest payload" },
    "search_space": {},
    "objective": { "metric": "sharpe", "direction": "max" },
    "budget": { "max_trials": 50 }
  }
}
```

## Explicit guardrails
- Forbidden values:
  - `spec_type=optimization`
  - `objective.direction=maximize|minimize`
  - stringified JSON in `search_space`
  - `low/high` keys for ranges (must be `min/max`)
- Disable submit when optimization is ON and form invalid.
- Keep optimization state local to current strategy screen (no bleed between DCA/Backtest forms).

## Implementation plan
1. Add optimization controls to existing DCA/Backtest reactive forms.
2. Add helper to derive submit mode from toggle state.
3. Keep existing DCA/Backtest payload builders for base spec.
4. When optimization is enabled, wrap base spec into canonical optimization request:
  - `optimize_dca` + `optimization.base_spec=<dca payload>`
  - `optimize_backtest` + `optimization.base_spec=<backtest payload>`
5. Wire capabilities fetch for `optimize_dca` / `optimize_backtest` and show runtime rules in UI hints.
6. Add UI-level smoke assertions in launcher spec for each submit mode:
  - DCA OFF -> `dca`
  - DCA ON -> `optimize_dca`
  - Backtest OFF -> `backtest`
  - Backtest ON -> `optimize_backtest`

## Non-goals / Out of scope
- New standalone optimization page or launcher tab.
- Backend contract changes.
