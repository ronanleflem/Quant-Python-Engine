# INIT-003 - Python backtest signal parity expansion

## Title
- Etendre la parite des signaux backtest en canonical

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
- Reduire les cas `accepted_but_not_wired` sur `signal` pour backtest canonical afin de fermer les gaps ta4j prioritaires.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/api/app.py`
  - `src/quant_engine/backtest/runner.py`
  - modules signaux backtest
- Pipeline integration points:
  - `/runs` submit
  - worker/result lifecycle
- Related docs:
  - ticket audit INIT-003 python

## BMAD Handover In
- Contrat d'erreur canonical gele.
- Priorites signaux legacy a supporter confirmees.

## BMAD Handover Out
- Nouveaux signaux backtest cables ou explicitement refuses en contrat.
- Reduction mesuree des echec `not_implemented_feature` sur `signal`.

## Context7 Decision
- Required: No
- Reason: implementation dans moteur Python interne.

## Constraints & conventions
- Preserve deterministic behavior unless specified otherwise.
- Prefer vectorized numpy/pandas operations.
- Avoid silent API/contract changes.

## Definition of Done
- [ ] Support runtime ajoute pour lot prioritaire de signaux backtest.
- [ ] Cas non supportes restants explicitement classes.
- [ ] Tests API + runner mis a jour.
- [ ] Validation commands pass.

## Implementation plan
1. Prioriser 1-2 signaux legacy a forte valeur.
2. Coder wiring canonical vers runner.
3. Ajouter tests contractuels et d'execution.

## Tests
- Unit tests:
  - tests parsing/dispatch signaux
- Integration tests:
  - lifecycle `/runs` + resultat signal
- Performance sanity (if applicable):
  - non regression runtime basique

## Validation commands
- `poetry run pytest -q`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Refonte complete de tous les signaux legacy en un ticket.
- Changement API Spring.

## Notes / pitfalls
- Eviter de modifier la taxonomie d'erreurs sans coordination cross-repo.
