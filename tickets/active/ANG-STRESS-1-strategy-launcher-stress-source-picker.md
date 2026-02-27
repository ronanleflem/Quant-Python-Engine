## Title
- ANG-STRESS-1 - Add stress test source-run picker and submit flow in strategy-launcher

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Cross-Repo Coordination
- Cross-Repo Initiative: Stress Test Launcher
- Repo Owner: angular-financial-project
- Upstream Dependencies:
  - SPR-RUNS-STRESS-1 (`GET /api/runs/stress/sources`)
  - ANG-STRESS-0 (adapter/model contract migration to `data.baseRunId`)
  - Python `GET /runs/capabilities?spec_type=stress_tests`
- Contract Version: catalog_version=2026-02-02

## Goal
- In `/strategy-launcher`, allow user to select an existing DCA/backtest run and submit a standalone stress test (`spec_type=stress_tests`).

## Audit findings (current state)
- Strategy launcher stress form is still built around `symbol/timeframe/startDate/endDate` and not around `baseRunId`.
- Strategy launcher loads capabilities for `dca/backtest/market_stats/seasonality` only; no `stress_tests` capability fetch.
- A dedicated stress page exists (`/stress-tests`) but it displays already computed stress runs/results, not source-run selection for submit.

## Context / Entry points
- Modules/files:
  - `src/app/pages/strategy-launcher/strategy-launcher.page.ts`
  - `src/app/pages/strategy-launcher/strategy-launcher.page.html`
  - `src/app/services/runs.service.ts`
  - `src/app/pages/stress-tests/stress-tests.page.ts` (reference/reuse only)
- Pipeline integration points:
  - `GET /api/runs/stress/sources`
  - `POST /api/runs`

## UX scope
1. New section/tab: `Stress Test`
2. Source run selector:
  - searchable list (run_id, type, symbol, timeframe, date)
  - filters: `spec_type`, date range
3. Stress params form:
  - driven by capabilities + catalog for `performance.stress_tests.*`
4. Payload preview + submit button

## Submit payload contract
```json
{
  "spec_type": "stress_tests",
  "catalog_version": "2026-02-02",
  "data": {
    "base_run_id": "<selected_run_id>"
  },
  "performance": {
    "stress_tests": {
      "enabled": true,
      "method": "bootstrap",
      "n_sims": 500,
      "seed": 42,
      "block_size": 5,
      "scenarios": []
    }
  }
}
```

## Definition of Done
- [ ] Strategy launcher has a dedicated stress test mode.
- [ ] User can select an eligible base run from backend list.
- [ ] Form validates required fields before submit.
- [ ] Payload uses `data.base_run_id` (no legacy symbol/timeframe fields).
- [ ] Run status/result screen supports `spec_type=stress_tests` and shows Monte Carlo/scenario blocks.
- [ ] Unit tests on serializer + form validation.

## Implementation plan
1. Add `RunsService.getStressSources(...)` client + DTO mapping.
2. Add source-run picker in stress tab (autocomplete/select + basic filters).
3. Replace stress “data” inputs in launcher by selected source run binding.
4. Fetch/apply `/runs/capabilities?spec_type=stress_tests` for stress form gating/tooltips.
5. Update payload preview and submit flow for canonical `data.base_run_id`.
6. Add tests (launcher submit payload + validation + picker behavior).

## Non-goals / Out of scope
- Editing/replaying base strategy run.
- Multi-run batch stress launch.
