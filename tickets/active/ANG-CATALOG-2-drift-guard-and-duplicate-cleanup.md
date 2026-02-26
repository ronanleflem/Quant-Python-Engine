## Title
- ANG-CATALOG-2 - Remove catalog duplication and add drift guard for public/docs

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Cross-Repo Coordination
- Cross-Repo Initiative: N/A
- Repo Owner: angular-financial-project
- Upstream Dependencies: ANG-CATALOG-1
- Contract Version: catalog_version=2026-02-02

## Goal
- Prevent future drift between `docs/parameter_catalog.json`, `public/parameter_catalog.json`, and legacy copies.

## Clear file policy (decision)
- Runtime file used by Angular app:
  - Keep `public/parameter_catalog.json` (served at `/parameter_catalog.json`).
- Legacy duplicate:
  - Delete `public/parameter_catalog_old.json`.
- Documentation copy:
  - Keep `docs/parameter_catalog.json` only if it is auto-synced from the runtime file (or vice-versa).
  - If no sync is implemented, remove `docs/parameter_catalog.json` to avoid divergence.

## Context / Entry points
- Modules/files:
  - `public/parameter_catalog.json`
  - `docs/parameter_catalog.json`
  - `public/parameter_catalog_old.json`
  - package scripts / CI workflow

## Definition of Done
- [ ] Remove `public/parameter_catalog_old.json`.
- [ ] Declare single source of truth:
  - either `public/parameter_catalog.json`
  - or `docs/parameter_catalog.json` with generation to `public`.
- [ ] Add guard script in CI:
  - fail if duplicate catalogs diverge.
- [ ] Document exact update flow in README/dev docs.

## Implementation plan
1. Confirm source-of-truth strategy:
  - Recommended now: `public/parameter_catalog.json` as canonical runtime source.
2. Remove `public/parameter_catalog_old.json`.
3. If `docs/parameter_catalog.json` is kept:
  - add sync command (`sync-catalog`) and CI check (`check-catalog-drift`).
4. Add npm script + CI step and fail build on drift.
5. Update docs with one-line rule: "Edit only <source-of-truth file>".

## Non-goals / Out of scope
- UI logic changes.
- Backend contract changes.
