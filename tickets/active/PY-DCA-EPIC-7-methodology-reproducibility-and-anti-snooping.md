## Title
- PY-DCA-EPIC-7 — Validation méthodologique, reproductibilité et anti data-snooping

## Ticket type
- Type A: Audit/Discovery

## BMAD Stage
- Architect

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-DCA-GRID-001
- Repo Owner: quant-python-engine
- Upstream Dependencies: PY-DCA-EPIC-1..6
- Contract Version: dca-grid-process-v1

## Goal
- Produire un cadre d’audit vérifiable: règles ex-ante gelées, traçabilité complète, contrôle de fuite d’information.

## Context / Entry points
- Modules/files:
  - `docs/dca_grid_implementation_process.md`
  - `docs/tests_plan.md`
  - `docs/canonical_runs_dca_source_of_truth_2026-02-20.md`

## Definition of Done
- [ ] Checklist anti-snooping prête pour revue.
- [ ] Matrice risques méthodologiques + mitigations.
- [ ] Procédure de rerun reproductible documentée.
- [ ] Critères go/no-go formalisés.

## Implementation plan
1. Formaliser checklist et protocoles de run.
2. Définir artefacts obligatoires pour audit trail.
3. Créer template de rapport méthodologique.

## Tests
- Unit tests:
  - N/A (ticket d’audit)
- Integration tests:
  - Rejeu d’un run canonique et comparaison artefacts.

## Validation commands
- `poetry run qe run-local --spec specs/examples/strategy_dca_etf_delta_2024_2026.json`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Développement de dashboards Angular.
