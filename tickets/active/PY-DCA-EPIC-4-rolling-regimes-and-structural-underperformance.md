## Title
- PY-DCA-EPIC-4 — Rolling windows, régimes de marché et sous-performance structurelle

## Ticket type
- Type B: Implementation

## BMAD Stage
- Architect

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-DCA-GRID-001
- Repo Owner: quant-python-engine
- Upstream Dependencies: PY-DCA-EPIC-2
- Contract Version: dca-grid-process-v1

## Goal
- Mesurer stabilité temporelle (3/5/10 ans), IRR glissant et périodes de sous-performance structurelle.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/validate/splitter.py`
  - `src/quant_engine/strategies/runner.py`
  - `src/quant_engine/persistence/models.py`

## Definition of Done
- [ ] Fenêtres glissantes paramétrables (3/5/10 ans) avec outputs indexés temporellement.
- [ ] Étiquetage de régime de marché documenté.
- [ ] Indicateur de sous-performance structurelle (durée + sévérité).

## Implementation plan
1. Étendre splitters pour rolling windows DCA.
2. Calculer métriques par fenêtre + régime.
3. Exporter séries prêtes pour visualisation.

## Tests
- Unit tests:
  - bordures de fenêtres, overlap, timezone.
- Integration tests:
  - run complet rolling sur dataset court.

## Validation commands
- `poetry run pytest -q tests/validate/test_dca_rolling_windows.py`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Définition finale du score composite.
