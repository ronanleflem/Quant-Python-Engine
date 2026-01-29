# EX-001 - Add Indicator With Tests (ATR-like Volatility)

## Title
Add ATR-like volatility indicator to feature pipeline with tests

## Goal
Add a new ATR-like volatility indicator that can be computed from OHLC data and used by the existing feature pipeline. The indicator must be vectorized, deterministic, and covered by unit + performance sanity tests.

## Context / Entry points
- `src/quant_engine/core/features/atr.py` (existing style to mirror)
- `src/quant_engine/core/features/store.py` (caching pattern)
- `src/quant_engine/core/features/__init__.py` (export if used)
- `src/quant_engine/stats/runner.py` (feature integration point)
- `tests/test_features.py` (unit tests for feature outputs)
- `tests/test_large_dataset_perf.py` (perf patterns)

## Constraints & conventions
- Vectorized pandas/numpy; avoid Python loops over rows.
- Output must align 1:1 with input rows.
- Deterministic, no randomness.
- Follow existing naming: `compute(...)` returning `List[float]` or `pd.Series` consistent with other features.
- Handle short series (`len < window`) by returning `None` / `nan` per project convention.

## Definition of Done (DoD)
- [ ] New indicator module added and imported where needed.
- [ ] Feature is computed via the pipeline (stats runner or feature store).
- [ ] Unit tests cover normal and edge cases (NaNs, short series).
- [ ] Performance sanity test added (large dataset timing).
- [ ] All validation commands pass.

## Implementation plan
1. Add `src/quant_engine/core/features/volatility_atr.py` with `compute(dataset, params)`.
2. Update `src/quant_engine/core/features/__init__.py` to export the new module if required.
3. Integrate into `src/quant_engine/stats/runner.py` to populate a `volatility_atr` column in stats output (optional feature flag in params).
4. Add unit tests in `tests/test_features.py` for:
   - constant price series (ATR-like should be near zero)
   - series with NaNs (propagate or skip per convention)
   - short series (`len < window`)
5. Add a simple perf check (e.g., `tests/test_large_dataset_perf.py`) asserting runtime below a reasonable threshold.

## Tests
- Unit: `tests/test_features.py::test_volatility_atr_basic`
- Unit: `tests/test_features.py::test_volatility_atr_short_series`
- Perf (slow): `tests/test_large_dataset_perf.py::test_volatility_atr_perf`

## Fast tests vs Slow tests
- Fast: `tests/test_features.py` (unit-only).
- Slow: `tests/test_large_dataset_perf.py` (marked `-m slow`).

## Validation commands
- `poetry run pytest -q tests/test_features.py`
- `poetry run pytest -m slow -q tests/test_large_dataset_perf.py -k volatility_atr`

## Non-goals / Out of scope
- No changes to existing ATR behavior.
- No refactor of feature store or caching.
- No new API endpoints.

## Notes / pitfalls
- Ensure the output index aligns with input rows.
- Beware of division by zero when close is zero.
- Confirm how NaNs are handled in current feature implementations and match that behavior.
