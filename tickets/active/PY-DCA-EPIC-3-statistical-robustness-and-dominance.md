## Title
- PY-DCA-EPIC-3 — Robustesse statistique, percentiles et dominance

## Ticket type
- Type B: Implementation

## BMAD Stage
- Architect

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-DCA-GRID-001
- Repo Owner: quant-python-engine
- Upstream Dependencies: PY-DCA-EPIC-1, PY-DCA-EPIC-2
- Contract Version: dca-grid-process-v1

## Goal
- Quantifier la robustesse de la DCA Grid via distribution benchmark passive, percentile, dominance stochastique simplifiée et stress tests paramètres.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/stats/runner.py`
  - `src/quant_engine/stats/estimators.py`
  - `src/quant_engine/optimize/runner.py` (stress grid)
- Related docs:
  - `docs/tests_plan.md`
  - `docs/dca_grid_implementation_process.md`

## BMAD Handover In
- Métriques calculées par run et schéma stable.

## BMAD Handover Out
- Artefacts de distribution et tableaux de robustesse consommables Angular.

## Context7 Decision
- Required: No
- Reason: Méthodes connues, pas de dépendance externe bloquante.

## Definition of Done
- [ ] Monte Carlo calendrier avec seed traçable.
- [ ] Percentile DCA Grid vs distribution passive calculé et persisté.
- [ ] Test de dominance (perf/drawdown) documenté.
- [ ] Stress tests grille (param sweep borné) reproductibles.

## Implementation plan
1. Ajouter job de simulation passive (N runs) + agrégats quantiles.
2. Calculer percentiles et dominance sur sorties métriques.
3. Exposer résultats en artefacts JSON/Parquet versionnés.

## Tests
- Unit tests:
  - déterminisme seed, cohérence percentiles.
- Integration tests:
  - spec de simulation sur petit dataset.

## Validation commands
- `poetry run pytest -q tests/stats/test_dca_robustness.py`
- `poetry run pytest -q tests/integration/test_dca_robustness_pipeline.py`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Dashboard Angular.

## Notes / pitfalls
- Ne pas confondre analyses ex-post et règles ex-ante.
