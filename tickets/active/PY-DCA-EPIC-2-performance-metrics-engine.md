## Title
- PY-DCA-EPIC-2 — Moteur de métriques DCA (perf normalisée, TWR, IRR/XIRR, drawdown, recovery)

## Ticket type
- Type B: Implementation

## BMAD Stage
- Architect

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-DCA-GRID-001
- Repo Owner: quant-python-engine
- Upstream Dependencies: PY-DCA-EPIC-1
- Contract Version: dca-grid-process-v1

## Goal
- Construire un moteur de métriques robuste et réutilisable pour comparer DCA Grid et benchmarks passifs à capital investi comparable.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/backtest/metrics.py`
  - `src/quant_engine/strategies/runner.py`
  - `src/quant_engine/persistence/db.py` (persist métriques)
- Pipeline integration points:
  - Endpoints `/runs/{run_id}/metrics`
- Related docs:
  - `docs/backtest_metrics.md`
  - `docs/dca_grid_implementation_process.md`

## BMAD Handover In
- Sorties benchmark homogènes depuis EPIC-1.

## BMAD Handover Out
- API de métriques versionnée + tables/artefacts enrichis.

## Context7 Decision
- Required: No
- Reason: Algorithmes standards implémentables avec stack locale numpy/pandas/polars.

## Definition of Done
- [ ] `final_performance_normalized`, `twr`, `xirr`, `max_drawdown_on_contributed_capital`, `time_under_water` disponibles.
- [ ] Tests numériques sur datasets synthétiques avec tolérance définie.
- [ ] Rendu compatible API/CLI existants.
- [ ] Documentation des limites (cashflows irréguliers, solutions IRR multiples).

## Implementation plan
1. Étendre `backtest/metrics.py` avec signatures typées + docstrings.
2. Brancher calculs dans pipeline run + persistance des métriques.
3. Créer tests unitaires ciblés + golden cases.

## Tests
- Unit tests:
  - `tests/backtest/test_dca_metrics.py`
- Integration tests:
  - `tests/api/test_runs_metrics_dca.py`

## Validation commands
- `poetry run pytest -q tests/backtest/test_dca_metrics.py`
- `poetry run pytest -q tests/api/test_runs_metrics_dca.py`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Score composite final.

## Notes / pitfalls
- XIRR: gérer les non-convergences sans masquer l’erreur (statut explicite).
