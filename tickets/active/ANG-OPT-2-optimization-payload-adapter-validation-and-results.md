## Title
- ANG-OPT-2 - Serialize optimization from DCA/Backtest forms to canonical optimize_* and render results

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Cross-Repo Coordination
- Cross-Repo Initiative: Optimization Launcher
- Repo Owner: angular-financial-project
- Upstream Dependencies:
  - PY-OPT-1
  - PY-OPT-2
  - ANG-OPT-1
- Contract Version: catalog_version=2026-02-02

## Goal
- Ensure optimization (enabled inside DCA/Backtest forms) is serialized to canonical Python contract and rendered in results.

## Context / Entry points
- Modules/files:
  - `src/app/models/run-request-input.model.ts`
  - `src/app/services/run-request-adapter.ts`
  - `src/app/pages/strategy-launcher/strategy-launcher.page.ts`
  - `src/app/pages/strategy-launcher/strategy-launcher.page.html`
  - `src/app/pages/strategy-launcher/strategy-launcher.page.spec.ts`
  - `src/app/services/runs.service.spec.ts`

## Definition of Done
- [ ] Adapter supports two optimization submit payloads only:
  - `spec_type=optimize_dca`
  - `spec_type=optimize_backtest`
- [ ] Generated optimization payload embeds baseline strategy payload in `optimization.base_spec`.
- [ ] Payload uses canonical keys and enums:
  - `objective.direction in {\"max\",\"min\"}`
  - search space object (not stringified JSON)
  - numeric ranges use `min`/`max` (not `low`/`high`)
- [ ] Validation blocks invalid optimization requests before submit:
  - missing objective
  - invalid/empty search space
  - `max_trials < 1`
- [ ] Result panel supports canonical optimization response shape:
  - objective summary
  - best params + best score
  - trial status split (`SUCCEEDED`/`FAILED`) and top trial rows
- [ ] Unit tests cover serializer + validation + result mapping.
- [ ] Adapter rejects/normalizes incompatible search-space shapes:
  - accepts: `values[]`, `domain[]`, `min/max(/step)`, plain array
  - rejects: empty arrays, invalid JSON, non-object root
- [ ] Adapter preserves existing DCA/backtest payload builder as source-of-truth for `optimization.base_spec`.

## Serialization rules (strict)
1. `objective.direction`:
  - accepted UI aliases: `maximize|minimize|max|min`
  - serialized canonical: `max|min`
2. `search_space`:
  - must be an object at submit time (never serialized as string)
  - if editor is text: parse JSON before payload build; fail fast on parse error
3. range keys:
  - canonical keys are `min|max|step`
  - transform UI `low|high` -> `min|max` before submit
4. budget:
  - `max_trials` required integer `>= 1`
  - `timeout_seconds` optional integer `>= 1`
  - `seed` optional integer `>= 0`

## Validation matrix (blocking)
- Missing objective metric => block submit
- Invalid direction enum => block submit
- `search_space` empty object => block submit
- Any search-space entry unresolved to supported shape => block submit
- `max_trials < 1` => block submit
- DCA screen + optimize ON + `base_spec.spec_type != dca` => block submit
- Backtest screen + optimize ON + `base_spec.spec_type != backtest` => block submit

## Result rendering contract
- Read from canonical response path:
  - `result.objective`
  - `result.best.score`
  - `result.best.params`
  - `result.trials[]`
  - `result.summary.total_trials|succeeded_trials|failed_trials`
- On failed optimization run (`status=FAILED`), show backend error code/message without silent fallback.

## Implementation plan
1. Add optimization request models for `optimize_dca` and `optimize_backtest`.
2. Extend `buildCanonicalRunPayload`:
  - if optimization toggle OFF => keep current `dca`/`backtest` payload
  - if optimization toggle ON => wrap base payload into `optimization.base_spec`
3. Normalize/validate search-space editor value into JSON object.
4. Enforce objective/budget/search-space validators before submit.
5. Render canonical optimization result blocks in launcher result panel.
6. Add tests in model, adapter and launcher spec files.
7. Add regression tests for invalid legacy payload patterns:
  - `spec_type=optimization`
  - `objective.direction=maximize`
  - `search_space` stringified JSON
  - `low/high` not converted.

## Validation commands
- `npm run test -- --include src/app/models/run-request-input.model.spec.ts`
- `npm run test -- --include src/app/services/run-request-adapter.spec.ts`
- `npm run test -- --include src/app/pages/strategy-launcher/strategy-launcher.page.spec.ts`

## Non-goals / Out of scope
- Historical optimization runs page.
- Backend changes.
