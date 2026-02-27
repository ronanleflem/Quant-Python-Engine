## Title
- SPR-RUNS-STRESS-1 - Expose selectable canonical runs for stress tests

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Cross-Repo Coordination
- Cross-Repo Initiative: Stress Test Launcher
- Repo Owner: financial-project (Spring)
- Upstream Dependencies: Python canonical `/runs` + `/runs/{id}` + `/runs/{id}/result`
- Downstream Dependencies: Angular strategy-launcher stress test UI
- Contract Version: catalog_version=2026-02-02

## Goal
- Provide a Spring endpoint that lists eligible source runs (DCA/backtest succeeded) so Angular can let users pick one and launch `spec_type=stress_tests`.

## Audit findings (current state)
- `RunController` exposes only `/api/runs*` proxy + capabilities, no source-run listing endpoint.
- `StressTestResultController` already has `/api/stress-tests/runs` but it lists runs that already have stress results, not eligible DCA/backtest source runs.
- `ApiJobEntity` + `ApiJobRepository` exist but are not used to expose filtered canonical source runs.

## Context / Entry points
- Modules/files:
  - `src/main/java/finance/project/api/controllers/RunController.java`
  - `src/main/java/finance/project/api/controllers/StressTestResultController.java`
  - `src/main/java/finance/project/api/entities/quant/ApiJobEntity.java`
  - `src/main/java/finance/project/api/repositories/ApiJobRepository.java`
  - new service + DTOs for source-run listing
- Pipeline integration points:
  - `GET /api/runs/stress/sources`
  - existing `POST /api/runs` proxy (for final stress test submit)

## API contract (proposed)
1. `GET /api/runs/stress/sources?limit=20&cursor=<optional>&strategyType=<optional:dca|backtest>`
2. Response item fields:
  - `run_id`
  - `spec_type`
  - `status`
  - `created_at`
  - `finished_at`
  - `symbol` (nullable)
  - `timeframe` (nullable)
  - `asset_class` (nullable)
  - `currency` (nullable)
  - `trades_count_estimate` (nullable)
3. Eligibility rule:
  - canonical run only
  - `status == SUCCEEDED`
  - `spec_type in ["dca", "backtest"]`

## Definition of Done
- [ ] New endpoint returns only eligible runs.
- [ ] Pagination supported (`limit`, `cursor`) with deterministic ordering (`finished_at desc, run_id desc`).
- [ ] DTOs include enough metadata for UI label and filtering.
- [ ] Endpoint covered by unit/integration tests.
- [ ] OpenAPI updated.

## Implementation plan
1. Add repository query methods on `ApiJobRepository` for canonical run jobs (type/status ordering).
2. Add service that parses `payload_json`/`result_json` to extract `spec_type`, symbol, timeframe, and optional trade count.
3. Expose `GET /api/runs/stress/sources` from `RunController` (or dedicated runs-stress controller).
4. Keep existing `/api/stress-tests/runs` unchanged (that endpoint remains “already computed stress runs”).
5. Add tests: controller + service parsing + pagination ordering.

## Non-goals / Out of scope
- Launching stress tests from this endpoint.
- Cancel flow changes.
