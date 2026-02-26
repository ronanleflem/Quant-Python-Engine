## Title
- PY-STATS-1 - Validate market_stats params and return 422 before worker execution

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Cross-Repo Coordination
- Cross-Repo Initiative: N/A
- Repo Owner: quant-python-engine
- Upstream Dependencies: None
- Contract Version: catalog_version=2026-02-02

## Goal
- Prevent runtime `execution_error` for invalid `stats.*.params` by validating required params in `/runs` and returning normalized HTTP 422 errors.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/api/run_request_input.py`
  - `src/quant_engine/api/app.py`
  - `src/quant_engine/stats/events.py`
  - `src/quant_engine/stats/conditions.py`
  - `src/quant_engine/stats/targets.py`
- Pipeline integration points:
  - canonical `/runs` payload validation for `spec_type=market_stats`
- Related docs:
  - `public/parameter_catalog.json` (Angular)

## Context7 Decision
- Required: No
- Reason: internal codebase contract and validation only.

## Constraints & conventions
- Do not change stats computation semantics.
- Keep existing normalized validation error format.

## Definition of Done
- [ ] Validation rejects invalid params with 422 (no worker retries).
- [ ] Required params enforced for key items.
- [ ] Unit/integration tests added for 422 errors.

## Implementation plan
1. Add canonical param validation for selected IDs:
   - event `k_consecutive`: `k>=1`, `direction in {up,down}`
   - condition `htf_trend`: `tf_multiplier>=1`, `ema_period>=1`
   - target `continuation_n`: `n>=1`, `direction in {up,down}`
   - target `time_to_reversal`: `max_horizon>=1`
2. Raise `ApiValidationException` with `field/code/message` paths under `market_stats.stats.*`.
3. Add tests in `tests/test_api_runs_submit_endpoint.py` (invalid payload -> 422).

## Validation commands
- `poetry run pytest -q tests/test_api_runs_submit_endpoint.py`

## Non-goals / Out of scope
- Exhaustive validation for every stats function parameter.
- UI-level changes.
