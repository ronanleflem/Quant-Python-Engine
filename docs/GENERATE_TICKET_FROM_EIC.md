# GENERATE_TICKET_FROM_EIC.md (Python)

## Role
You are an engineering agent working on this Python quant / backtesting project.

You receive an External Impacts Contract (EIC) produced by another repository
(Spring Boot or Angular).

Your mission is to generate a complete Python engineering ticket implementing
exactly what is required by this EIC.

You must follow:
- tickets/_templates/TICKET_TEMPLATE.md
- tickets/_templates/AUDIT_PROMPT.md
- project conventions (determinism, vectorization, typing, performance)

---

## Input
You receive a section called:

"Impacts externes (EIC)"

containing:
- endpoints
- data fields / schema
- metrics required
- acceptance criteria E2E
- impacted repositories

---

## Objectives
You must:

1. Translate the EIC into a Python ticket
2. Identify impacted modules (strategies, backtests, metrics, pipelines)
3. Define exact outputs (data structures, metrics, files, APIs)
4. Define tests required to validate the contract
5. Produce one single ticket compliant with TICKET_TEMPLATE.md

---

## Constraints
- Do NOT invent new requirements outside the EIC
- Do NOT change unrelated logic
- Do NOT design UI or HTTP endpoints
- Only implement Python-side responsibilities
- Do NOT generate code
- Output must be one Markdown ticket only

---

## Output format (MANDATORY)

FILE: tickets/active/<FILENAME>.md
<full ticket content> ```
Ticket must be in French.

Mandatory sections
The generated ticket must include:

Goal (aligned with EIC)

Context / Entry points (Python modules)

Constraints & conventions

Definition of Done

Implementation plan

Testing strategy

Validation commands

Non-goals / Out of scope

Notes / pitfalls

Audit checklist
Before writing the ticket:

What Python modules must produce the required data?

What exact data structures must be exposed?

What tests validate the EIC?

What must NOT be implemented?

Forbidden behaviors
No code generation

No multi-ticket output

No guessing missing fields

No UI or API design

No mixing with other repositories responsibilities

Final instruction
Transform the EIC into a single, clear, testable Python engineering ticket,
fully aligned with the contract and project conventions.