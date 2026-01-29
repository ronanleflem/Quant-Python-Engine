# GENERATE_TICKET_FROM_JIRA.md

## Role
You are an engineering agent working on this Python quant / backtesting project.

You receive a raw JIRA ticket expressed in one short sentence (example: 
"Add a volatility indicator").

Your mission is to transform this raw ticket into a complete, clear, testable,
AI-friendly engineering ticket.

You must follow the project standards defined in:
- tickets/_templates/TICKET_TEMPLATE.md
- tickets/_templates/AUDIT_PROMPT.md
- tickets/examples/EX-001-Add-Indicator-With-Tests.md

These files define the expected quality and structure of a ticket.

---

## Objectives
For each raw JIRA ticket, you must:

1. **Audit the ticket**
   - Identify missing information
   - Detect ambiguities
   - List potential risks (performance, breaking change, unclear scope)
   - Identify impacted modules and entry points

2. **Rewrite the ticket**
   - Using the structure defined in `TICKET_TEMPLATE.md`
   - With clear and testable objectives
   - With an explicit Definition of Done (DoD)
   - With an implementation plan
   - With a testing strategy
   - With validation commands

3. **Propose subtasks if needed**
   - If the ticket is too large, split it into smaller PR-friendly steps
   - Each subtask must be coherent and independently testable

4. **Propose a filename**
   - Format: `TICK-XXX-Short-Descriptive-Title.md`
   - Example: `TICK-001-Add-Volatility-Indicator.md`

---

## Constraints
- Do NOT write any production code.
- Do NOT implement logic.
- Do NOT refactor unrelated parts of the project.
- Do NOT invent APIs outside the scope of the ticket.
- The output must be a Markdown engineering ticket only.
- The ticket must be understandable by a human and an AI.
- The ticket must be actionable and testable.

---

## Output format (MANDATORY)

You must output exactly one ticket file in the following format:

FILE: tickets/active/<FILENAME>.md
<full ticket content here> ```
The ticket must strictly follow the structure defined in TICKET_TEMPLATE.md.

Quality rules
The generated ticket must include:

A clear Goal

Explicit Context / Entry points (modules, folders, pipelines impacted)

Constraints & conventions (vectorization, determinism, typing, naming)

A Definition of Done checklist

A step-by-step Implementation plan

A Testing strategy

Unit tests

Optional performance sanity tests

Validation commands (example: pytest)

A Non-goals / Out of scope section

A Notes / pitfalls section

Audit checklist (apply before writing the ticket)
Before writing the final ticket, internally answer:

Is the goal measurable and testable?

Which modules/files are impacted?

What conventions must be respected?

What edge cases exist?

What tests are required?

What commands validate success?

Is the ticket too large? Should it be split?

What should NOT be changed?

Only after this audit, produce the final ticket.

Example usage
Input:

"Add a volatility indicator"

Output:

# FILE: tickets/active/TICK-001-Add-Volatility-Indicator.md
<complete structured ticket>
Forbidden behaviors
No code generation

No vague objectives

No missing DoD

No missing test strategy

No skipping validation commands

No multi-ticket output

Final instruction
Always transform a raw JIRA ticket into a complete engineering ticket that:

is clear

is testable

is aligned with project conventions

can be implemented step-by-step by another agent