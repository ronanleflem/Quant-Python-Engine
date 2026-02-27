## Title
- ANG-STRESS-0 - Migrate stress_tests model/adapter contract to baseRunId

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Cross-Repo Coordination
- Cross-Repo Initiative: Stress Test Launcher
- Repo Owner: angular-financial-project
- Upstream Dependencies: Python canonical `/runs` (stress_tests contract)
- Contract Version: catalog_version=2026-02-02

## Goal
- Align Angular core payload model and serializer with canonical stress-tests contract: `data.base_run_id` + `performance.stress_tests`.

## Audit findings (current state)
- `buildCanonicalRunPayload` blocks submissions when `data.symbol` is absent (global guard), which breaks `stress_tests` independent flow.
- `RunRequestInput` stress-tests type still uses period/symbol fields instead of source run reference.
- Local validator enforces `data.symbol/timeframe/startDate/endDate` for `runType='stress_tests'`.
- Existing tests still assert old payload shape for stress tests.

## Context / Entry points
- Files:
  - `src/app/services/run-request-adapter.ts`
  - `src/app/models/run-request-input.model.ts`
  - `src/app/services/runs.service.spec.ts`
  - `src/app/pages/strategy-launcher/strategy-launcher.page.spec.ts` (stress assertions)

## Definition of Done
- [ ] Stress-tests input model uses `data.baseRunId` (UI) -> canonical `data.base_run_id` (API).
- [ ] Adapter symbol hard-guard no longer applies to `runType='stress_tests'`.
- [ ] Validator requires `data.baseRunId` and no longer requires symbol/timeframe/date range for stress-tests.
- [ ] Unit tests updated for new stress payload contract.

## Implementation plan
1. Add `baseRunId` to stress-tests data model type.
2. Special-case adapter prevalidation for stress-tests.
3. Update `validateRunRequest` branch for `stress_tests` required fields.
4. Update service/component tests that assert stress payload.

## Non-goals / Out of scope
- UI source-run picker itself (covered by ANG-STRESS-1).
- Backend endpoint creation.
