# INIT-003 - Python DCA grid preset parity

## Title
- Cablage canonical des presets DCA grid legacy prioritaires

## Ticket type
- Type B: Implementation

## BMAD Stage
- PM

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-003
- Repo Owner: quant-python-engine
- Upstream Dependencies: INIT-003-spring-canonical-runs-contract-guardrails
- Contract Version: STRAT-CALC-DECOMMISSION-V1-2026-03-05

## Goal
- Convertir les presets DCA legacy les plus utilises (`grid_conservative`/`grid_aggressive`) en support canonical pour reduire les echec runtime non implementes.

## Context / Entry points
- Modules/files:
  - API capabilities/signaling
  - strategie DCA/grid mapping
- Pipeline integration points:
  - `/runs` canonical DCA submit
  - worker execution DCA
- Related docs:
  - ticket audit INIT-003 python

## BMAD Handover In
- Guardrails Spring sur transport canonical.
- Matrice capabilities existante avec gaps DCA identifies.

## BMAD Handover Out
- Presets DCA prioritaires supportes en canonical.
- Capabilities mises a jour et alignees avec runtime.

## Context7 Decision
- Required: No
- Reason: scope purement interne Python.

## Constraints & conventions
- Preserve deterministic behavior unless specified otherwise.
- Prefer vectorized numpy/pandas operations.
- Avoid silent API/contract changes.

## Definition of Done
- [ ] Presets cibles cables et testes.
- [ ] `/runs/capabilities` aligne avec support effectif.
- [ ] Cas restants documentes en `not_implemented_feature`.
- [ ] Validation commands pass.

## Implementation plan
1. Mapper presets legacy vers configuration canonical DCA.
2. Implementer execution/validation.
3. Mettre a jour tests capabilities + runtime.

## Tests
- Unit tests:
  - mapping presets DCA
- Integration tests:
  - submit/run/result DCA sur presets cibles
- Performance sanity (if applicable):
  - check basique temps/ressources

## Validation commands
- `poetry run pytest -q`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Refonte globale du moteur DCA.
- Modifications Spring/Angular hors contrat.

## Notes / pitfalls
- Garder la semantique presets stable pour compat front.
