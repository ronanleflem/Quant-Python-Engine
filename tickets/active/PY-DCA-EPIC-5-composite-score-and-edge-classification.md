## Title
- PY-DCA-EPIC-5 — Score composite DCA et classification d’edge

## Ticket type
- Type B: Implementation

## BMAD Stage
- Architect

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-DCA-GRID-001
- Repo Owner: quant-python-engine
- Upstream Dependencies: PY-DCA-EPIC-2, PY-DCA-EPIC-3, PY-DCA-EPIC-4
- Contract Version: dca-grid-process-v1

## Goal
- Produire un score composite interprétable (performance, IRR, drawdown, robustesse) et un niveau d’edge (faible/moyen/fort).

## Context / Entry points
- Modules/files:
  - `src/quant_engine/backtest/metrics.py`
  - `src/quant_engine/io/`
  - `src/quant_engine/api/app.py`

## Definition of Done
- [ ] Formule et poids versionnés.
- [ ] Score décomposable par composante.
- [ ] Mapping vers classes d’edge documenté.
- [ ] Sensibilité du score aux poids reportée.

## Implementation plan
1. Ajouter module de scoring avec config de pondération.
2. Calculer score + composantes et persister.
3. Exposer via API et artefacts.

## Tests
- Unit tests:
  - invariants (borne, monotonie locale).
- Integration tests:
  - présence score dans résultat run.

## Validation commands
- `poetry run pytest -q tests/backtest/test_dca_score.py`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Design UI du score.
