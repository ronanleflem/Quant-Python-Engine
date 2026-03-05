## Title
- PY-MI-EPIC-3 — Intégration MI dans backtest, DCA et stress tests

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Goal
- Faire consommer des features MI pré-calculées aux moteurs sans embarquer la logique de calcul dans les stratégies.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/backtest/`
  - `src/quant_engine/strategies/`
  - `src/quant_engine/performance/stress_tests.py`

## Tickets enfants
- PY-MI-3.1: Injection de dépendance `MarketIntelligenceService` dans run backtest.
- PY-MI-3.2: Injection MI dans run stratégie/DCA (feature row fournie à la stratégie).
- PY-MI-3.3: Ajouter mode fallback legacy (MI OFF) via adapter.
- PY-MI-3.4: Étendre stress tests pour scénarios de shifts de régime + vol spikes en utilisant labels MI.
- PY-MI-3.5: Tests d’intégration backtest/DCA avec MI ON vs OFF.

## Definition of Done
- [ ] Backtest et DCA exécutent avec features MI sans régression majeure.
- [ ] Stress tests consomment labels régime de manière contractuelle.
- [ ] Tests d’intégration verts.

## Validation commands
- `poetry run pytest -q tests/backtest tests/strategies tests/integration`

## Notes / pitfalls
- Garder compatibilité specs existantes.
- Éviter d’ajouter des imports `market_intelligence` profonds dans les stratégies.
