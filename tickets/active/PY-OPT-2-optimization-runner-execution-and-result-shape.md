## Title
- PY-OPT-2 - Execute canonical optimization runs and return structured best/trials results

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Cross-Repo Coordination
- Cross-Repo Initiative: Optimization Launcher
- Repo Owner: quant-python-engine
- Upstream Dependencies:
  - PY-OPT-1
- Downstream Dependencies:
  - ANG-OPT-2
- Contract Version: catalog_version=2026-02-02

## Goal
- Wire optimization execution in canonical `/runs` worker flow for backtest and DCA targets.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/api/app.py`
  - `src/quant_engine/optimize/runner.py`
  - `src/quant_engine/strategies/runner.py` (if shared execution helpers are needed)
- Pipeline integration points:
  - worker processing path for canonical run requests
  - `/runs/{id}/result`

## Expected result shape
```json
{
  "accepted": true,
  "spec_type": "optimize_backtest",
  "result": {
    "objective": { "metric": "sharpe", "direction": "max" },
    "best": {
      "score": 1.23,
      "params": { "signal.fast": 10, "signal.slow": 30 },
      "run_id": "..."
    },
    "trials": [
      { "trial_id": 1, "score": 1.02, "status": "SUCCEEDED", "params": { ... } }
    ],
    "summary": {
      "total_trials": 50,
      "succeeded_trials": 48,
      "failed_trials": 2
    }
  }
}
```

## Definition of Done
- [x] Optimization runs execute through canonical worker path.
- [x] Supports backtest and DCA target modes.
- [x] Returns deterministic structured payload (`best`, `trials`, `summary`).
- [x] Handles partial failures per trial without crashing whole run.
- [x] Tests added for success/failure/mixed outcomes.

## Implementation plan
1. Add optimization handling branch in canonical job execution (`app.py`).
2. Adapt or wrap `optimize.runner.run` for canonical input/output.
3. Normalize per-trial status and scoring.
4. Persist result in existing job result JSON.
5. Add regression tests for run lifecycle and result schema.

## Validation commands
- `poetry run pytest -q tests/test_api_runs_worker.py`
- `poetry run pytest -q tests/test_api_runs_lifecycle_endpoints.py`

## Implementation status
- Canonical optimization execution is wired in `JOB_TYPE_CANONICAL_RUN` for:
  - `spec_type=optimize_backtest`
  - `spec_type=optimize_dca`
- Canonical payload is adapted to `optimize.variants` runners:
  - backtest target -> `run_backtest_optimization`
  - dca target -> `run_strategy_optimization`
- `optimization.base_run_id` now resolves and validates canonical base run payload/status before execution.
- Structured result now includes:
  - `objective`
  - `best`
  - `trials` (normalized status/score)
  - `summary` (total/succeeded/failed)
  - `source` + artifacts paths.
- `/runs/capabilities` runtime matrix updated for optimize specs (`execution_status: wired`).
- Tests updated for:
  - success payload
  - mixed trial outcomes
  - validation failure on base run type mismatch.

## Non-goals / Out of scope
- Dedicated SQL tables for trials history.
- Angular UI rendering.
