# AUDIT PROMPT (Ticket Quality Checklist)

## Objective clarity
- [ ] The goal is testable and observable (no vague outcomes).
- [ ] Success criteria are measurable (output columns, metrics, files, logs).

## Scope & files
- [ ] Impacted files/modules are listed (paths).
- [ ] Entry points and integration points are named.
- [ ] Non-goals are explicit.

## Conventions & constraints
- [ ] Project naming conventions are stated.
- [ ] Performance expectations are stated (vectorization, no heavy loops).
- [ ] Determinism requirements are stated.
- [ ] Data assumptions (index, freq, timezone, columns) are stated.

## Edge cases
- [ ] Edge cases and failure modes listed (NaNs, short series, missing data).
- [ ] Backward compatibility risks identified.

## Test strategy
- [ ] Unit tests are specified (what and where).
- [ ] Integration tests are specified (pipeline or dataset).
- [ ] Performance sanity test is specified (simple timing).
- [ ] Fast vs slow tests are identified.

## Validation commands
- [ ] Concrete commands are provided (pytest, lint, etc.).
- [ ] Commands use Poetry when applicable (`poetry run pytest ...`).
- [ ] Any required environment variables are listed.

## Risk management
- [ ] Performance risk assessed.
- [ ] Breaking change risk assessed.
- [ ] Data quality / correctness risks noted.

## PR-friendly decomposition
- [ ] Ticket can be split into small steps or sub-tasks.
- [ ] Each step is reviewable and testable on its own.

## Agent constraints (Do NOT do)
- [ ] No large refactors outside the ticket scope.
- [ ] No silent API changes (must be documented in ticket).
- [ ] No changes to unrelated modules or tests.
- [ ] No weakening of tests or skipping validations.
