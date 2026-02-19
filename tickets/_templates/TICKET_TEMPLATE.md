# FILE: tickets/_templates/TICKET_TEMPLATE.md
# Ticket Template (Python)

## Title
- [Short, actionable title]

## Ticket type
- [Type A: Audit/Discovery | Type B: Implementation]

## BMAD Stage
- [PM | Architect | Dev | Reviewer]

## Cross-Repo Coordination
- Cross-Repo Initiative: [INIT-xxx or N/A]
- Repo Owner: [quant-python-engine]
- Upstream Dependencies: [ticket/PR ids or None]
- Contract Version: [version/tag/commit or N/A]

## Goal
- [Observable and testable outcome]

## Context / Entry points
- Modules/files:
- Pipeline integration points:
- Related docs:

## BMAD Handover In
- [Required artifacts from previous stage]

## BMAD Handover Out
- [Artifacts produced for next stage]

## Context7 Decision
- Required: [Yes/No]
- Reason: [One short justification]

## Constraints & conventions
- Preserve deterministic behavior unless specified otherwise.
- Prefer vectorized numpy/pandas operations.
- Avoid silent API/contract changes.

## Definition of Done
- [ ] Feature implemented per goal.
- [ ] Unit tests added or updated.
- [ ] Integration tests added or updated.
- [ ] Validation commands pass.

## Implementation plan
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Tests
- Unit tests:
- Integration tests:
- Performance sanity (if applicable):

## Validation commands
- `poetry run pytest -q`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- [Explicit list]

## Notes / pitfalls
- [Edge cases, determinism, data assumptions]
