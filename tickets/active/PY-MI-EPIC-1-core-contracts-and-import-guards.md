## Title
- PY-MI-EPIC-1 — Contrats core et garde-fous d’architecture

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Goal
- Poser les interfaces stables (`core/domain`, `core/contracts`) et verrouiller les dépendances interdites dès le début.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/core/`
  - `tests/architecture/`
  - `docs/architecture_market_intelligence_refactor.md`

## Tickets enfants
- PY-MI-1.1: Créer `core/domain/models.py` (`Candle`, `Trade`, `Position`, `Portfolio`).
- PY-MI-1.2: Créer `core/contracts/market_intelligence.py`, `feature_store.py`, `strategy.py`.
- PY-MI-1.3: Ajouter `tests/architecture/test_import_rules.py` (imports interdits).
- PY-MI-1.4: Ajouter adaptateurs de compatibilité minimale pour l’existant (sans breaking change).

## Definition of Done
- [ ] Contrats versionnés et importables.
- [ ] Test d’architecture en place et vert.
- [ ] Aucun changement de comportement métier observé sur les tests existants.

## Validation commands
- `poetry run pytest -q tests/architecture/test_import_rules.py`
- `poetry run pytest -q tests/backtest tests/strategies`

## Notes / pitfalls
- Ne pas coupler `core` vers `api`, `backtest`, `performance`, `optimize`.
