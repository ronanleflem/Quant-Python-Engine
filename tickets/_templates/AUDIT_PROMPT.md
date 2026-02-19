# FILE: tickets/_templates/AUDIT_PROMPT.md
# Ticket Audit Prompt (Python + BMAD)

## PM gate
- [ ] Goal is clear, measurable, and user-visible/pipeline-visible.
- [ ] Scope and non-goals are explicit.
- [ ] DoD is testable.

## Architect gate
- [ ] Impacted modules/integration points identified.
- [ ] Approach coherent with existing architecture.
- [ ] Risks and rollback strategy documented.

## Dev gate
- [ ] Work can be split into small reviewable steps.
- [ ] Test strategy and validation commands are explicit.
- [ ] Determinism and data assumptions are explicit.

## Reviewer gate
- [ ] Review criteria are explicit and blocking.
- [ ] Regression risks are identified.
- [ ] Acceptance can be decided from evidence.

## Cross-repo gate (if applicable)
- [ ] `Cross-Repo Initiative` is set (`INIT-xxx`).
- [ ] External dependencies are explicit.
- [ ] Contract/version reference is explicit and testable.
- [ ] Scope remains local to Python repo.

## Context7 check (required only if needed)
- [ ] New or uncertain external API/library/framework involved.
- [ ] Version-specific behavior may affect implementation.
- [ ] If no, Context7 is intentionally skipped.
