## Title
- PY-OPT-1 - Add canonical optimization spec_type contract and /runs/capabilities matrix

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Cross-Repo Coordination
- Cross-Repo Initiative: Optimization Launcher
- Repo Owner: quant-python-engine
- Upstream Dependencies: None
- Downstream Dependencies:
  - ANG-OPT-1
  - ANG-OPT-2
- Contract Version: catalog_version=2026-02-02

## Goal
- Introduce a canonical optimization contract in `/runs` so Angular can submit optimization jobs for backtest and DCA without using legacy `/submit`.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/api/run_request_input.py`
  - `src/quant_engine/api/app.py`
- Pipeline integration points:
  - `POST /runs` with new optimization `spec_type`
  - `GET /runs/capabilities?spec_type=<optimization>`

## Proposed contract
1. Introduce one of:
  - `spec_type=optimize` with `target_spec_type in {backtest,dca}`
  - or two explicit specs: `optimize_backtest` + `optimize_dca`
2. Required blocks:
  - `base_spec` (canonical backtest/dca-like baseline payload)
  - `objective` (`metric`, `direction`)
  - `budget` (`max_trials`, optional `timeout_seconds`, optional `seed`)
  - `search_space` (named params with type and domain/range)

## Definition of Done
- [x] New optimization `spec_type` validates through `run_request_input`.
- [x] Validation errors return normalized HTTP 422.
- [x] `/runs/capabilities` exposes supported optimization fields + enums.
- [x] `runtime_rules` clarify limits (`max_trials`, supported metrics, etc.).
- [x] Endpoint tests added/updated.

## Implementation plan
1. Add strict Pydantic models for optimization blocks in `run_request_input.py`.
2. Extend `RunRequestInput` union with optimization request type(s).
3. Add capabilities branch in `_runs_capabilities_for_spec_type`.
4. Add tests for:
  - valid payload accepted
  - invalid search space rejected with 422
  - capabilities payload includes optimization fields

## Validation commands
- `poetry run pytest -q tests/test_api_runs_submit_endpoint.py`
- `poetry run pytest -q tests/test_api_runs_lifecycle_endpoints.py`

## Implementation status
- Implemented canonical spec types: `optimize_backtest`, `optimize_dca`.
- Added strict validation models for `optimization.objective`, `optimization.budget`, `optimization.search_space`, and `base_run_id|base_spec`.
- `/runs/capabilities` now returns optimization capability matrix + runtime rules for both optimize spec types.
- Canonical optimize jobs are accepted by `/runs` contract and currently return `not_implemented_feature` at execution time (explicitly documented behavior for PY-OPT-1 scope).

## Non-goals / Out of scope
- Executing optimization trials (handled in PY-OPT-2).
- Angular UI changes.
