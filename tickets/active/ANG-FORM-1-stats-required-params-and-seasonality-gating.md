## Title
- ANG-FORM-1 - Enforce required stats params and disable unsupported seasonality blocks

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Cross-Repo Coordination
- Cross-Repo Initiative: N/A
- Repo Owner: angular-financial-project
- Upstream Dependencies: PY-STATS-1, PY-SEASONALITY-1
- Contract Version: catalog_version=2026-02-02

## Goal
- Stop sending invalid payloads by enforcing required stats params in UI and clearly marking non-wired seasonality blocks.

## Context / Entry points
- Modules/files:
  - market stats form components
  - seasonality form components
  - run payload serializer
- Pipeline integration points:
  - `/api/runs` submit flow

## Definition of Done
- [ ] UI enforces required params for key stats ids:
  - `k_consecutive`: `k`, `direction`
  - `htf_trend`: `tf_multiplier`, `ema_period`
  - `continuation_n`: `n`, `direction`
  - `time_to_reversal`: `max_horizon`
- [ ] Dynamic stats param controls use validators (required + `>=1` for integer spans/periods).
- [ ] Default value `0` is no longer auto-sent for required numeric stats params.
- [ ] Seasonality `execution/risk/tp_sl` marked as "accepted but not wired" or disabled.
- [ ] Payload preview matches canonical schema exactly.

## Implementation plan
1. Extend catalog param mapping to preserve `required` metadata (currently dropped by `ParameterCatalogService.normalizeCatalogParams`).
2. In strategy launcher, apply validators on dynamic stats controls created by `ensureMarketStatsParamControls`.
3. Prevent serialization of invalid/empty required stats params in `buildMarketParams`.
4. Mark seasonality `execution` block as non-wired in UI copy/tooltip based on capabilities.
5. Add unit/E2E tests for invalid stats params and seasonality UX gating.

## Current state (observed)
- Dynamic stats params are loaded from `stats_expanded` and controls are created.
- But dynamic controls are created without validators and numeric defaults are `0` (`defaultParamValue`), which still causes backend 422 on required positive ints.

## Non-goals / Out of scope
- Implementing seasonality execution/risk logic backend.
