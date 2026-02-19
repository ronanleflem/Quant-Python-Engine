# FILE: docs/AI_WORKFLOW.md
# AI Workflow Guide (Python local execution)

This file defines local execution rules for Python only.

Global orchestration is managed in:
`C:\Users\ronan\Desktop\Cross-repo-coordination\Cross-repo-coordination`

## Local scope rules
- Keep scope local to Python repository.
- Do not orchestrate Spring or Angular from this file.
- Follow local ticket template and audit prompt.

## Local BMAD sequence
1. PM
2. Architect
3. Dev
4. Reviewer

## Context7
Use only when external docs are required.

## Validation
- `poetry run pytest -q`
