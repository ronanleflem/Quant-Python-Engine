# TICKET TEMPLATE

## Title
- Short, action-oriented, and specific (e.g., "Add ATR Indicator With Tests").

## Goal
- One or two sentences describing the user-visible or pipeline-visible outcome.
- Must be testable (describe observable behavior or artifact).

## Context / Entry points
- Files / modules likely to be touched (paths only).
- Where the new logic plugs into the pipeline (e.g., feature builder, indicator registry).
- Related specs or docs (paths only).

## Constraints & conventions
- Naming conventions (module/class/function).
- Performance expectations (vectorized numpy/pandas; avoid Python loops).
- Determinism rules (no randomness unless explicitly seeded).
- Data assumptions (index type, column names, timezone, frequency).

## Definition of Done (DoD)
- [ ] Feature implemented per Goal.
- [ ] Unit tests added and passing.
- [ ] Integration / pipeline tests updated or added.
- [ ] Performance sanity check added (if relevant).
- [ ] Validation commands executed successfully.
- [ ] Docs / examples updated if behavior changes.

## Implementation plan
1. Step-by-step plan with small, reviewable changes.
2. Include new files and updated files explicitly.
3. Mention any new config or fixtures.

## Tests
- Unit tests: target functions and edge cases.
- Integration tests: pipeline run or feature set validation.
- Performance sanity: quick timing or scale check.

## Fast tests vs Slow tests
- Fast tests: focused unit/integration tests that run quickly.
- Slow tests: marked or heavy tests (e.g., `-m slow`); run explicitly.
- Always include at least one fast test in the validation commands.

## Validation commands
- Exact commands to run locally (e.g., `poetry run pytest -q`).
- Any environment variables or optional flags.

## Non-goals / Out of scope
- Explicitly list what should NOT be touched.

## Notes / pitfalls
- Known edge cases, numeric stability issues, or API contracts to preserve.
