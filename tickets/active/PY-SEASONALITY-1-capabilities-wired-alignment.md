## Title
- PY-SEASONALITY-1 - Expose clear wired vs accepted fields in /runs/capabilities

## Ticket type
- Type B: Implementation

## BMAD Stage
- Done

## Status
- Done

## Cross-Repo Coordination
- Cross-Repo Initiative: N/A
- Repo Owner: quant-python-engine
- Upstream Dependencies: Angular capabilities consumer
- Contract Version: catalog_version=2026-02-02

## Goal
- Ensure `/runs/capabilities` communicates seasonality support accurately so frontend can disable unsupported blocks.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/api/app.py`
- Pipeline integration points:
  - `GET /runs/capabilities?spec_type=seasonality`

## Context7 Decision
- Required: No
- Reason: local API contract only.

## Definition of Done
- [x] `supported` includes canonical `data.start_date/end_date`.
- [x] `accepted_but_not_wired` explicitly lists `seasonality.execution/risk/tp_sl`.
- [x] runtime_rules text aligned with current behavior.
- [x] endpoint tests updated.

## Implementation plan
1. Review seasonality capability payload in `_runs_capabilities_for_spec_type`.
2. Align support lists with actual mapping code in `_canonical_seasonality_to_spec`.
3. Add/adjust tests in `tests/test_api_runs_lifecycle_endpoints.py`.

## Validation commands
- `poetry run pytest -q tests/test_api_runs_lifecycle_endpoints.py`

## Validation evidence
- Capabilities seasonality includes:
  - `data.start_date`, `data.end_date` in `fields.supported`
  - `seasonality.execution`, `seasonality.risk`, `seasonality.tp_sl` in `fields.supported` and `fields.accepted_but_not_wired`
- Regression test tightened:
  - `tests/test_api_runs_lifecycle_endpoints.py::test_runs_capabilities_returns_seasonality_runtime_matrix`

## Non-goals / Out of scope
- Implementing `execution/risk/tp_sl` logic in seasonality runner.
