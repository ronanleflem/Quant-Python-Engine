## Title
- PY-MI-EPIC-5 — Migration progressive, dépréciations et hardening final

## Ticket type
- Type B: Implementation

## BMAD Stage
- Reviewer

## Goal
- Finaliser la migration sans dette cachée: déprécier proprement le legacy, sécuriser la non-régression et documenter les garde-fous de dérive architecture.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/api/app.py`
  - `src/quant_engine/cli/main.py`
  - `docs/`
  - `tests/`

## Tickets enfants
- PY-MI-5.1: Dépréciations progressives (`warnings`, changelog, fenêtre de migration).
- PY-MI-5.2: Réduction progressive de `api/app.py` (handlers fins + services).
- PY-MI-5.3: Non-régression KPI (baseline specs backtest + DCA + stress).
- PY-MI-5.4: Durcir tests architecture (forbidden imports + smoke DI).
- PY-MI-5.5: Doc finale d’exploitation (`qe features recompute/inspect`, runbooks cache).

## Definition of Done
- [ ] Chemins legacy encore supportés mais balisés.
- [ ] Diff KPI dans tolérance documentée.
- [ ] Documentation d’exploitation à jour.

## Validation commands
- `poetry run pytest -q`

## Notes / pitfalls
- Toute rupture de contrat doit être versionnée et explicitée.
