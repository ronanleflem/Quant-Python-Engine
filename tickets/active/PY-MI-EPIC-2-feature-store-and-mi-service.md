## Title
- PY-MI-EPIC-2 — Feature Store + MarketIntelligenceService

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Goal
- Introduire la couche `market_intelligence` opérationnelle (v1) avec cache mémoire + disque et premiers labels (régime/liquidité/corrélation).

## Context / Entry points
- Modules/files:
  - `src/quant_engine/market_intelligence/` (nouveau)
  - `src/quant_engine/filters/` (adapters temporaires)
  - `src/quant_engine/stats/` (réutilisation partielle)

## Tickets enfants
- PY-MI-2.1: Créer `market_intelligence/models.py` (FeatureFrame, FeatureMeta, labels).
- PY-MI-2.2: Implémenter `FeatureStore` in-memory (`InMemoryFeatureStore`).
- PY-MI-2.3: Implémenter `FeatureStore` parquet (`ParquetFeatureStore`) avec clé `(feature_set, symbol, timeframe, version)`.
- PY-MI-2.4: Implémenter `MarketIntelligenceService` v1 avec pipelines:
  - corrélation glissante,
  - régime (trend/range/compression),
  - flags liquidité / magnet-failure.
- PY-MI-2.5: Tests unitaires sur shape/colonnes/index UTC/NaN policy.

## Definition of Done
- [ ] Service MI calcule et lit des features persistées.
- [ ] Colonnes de features documentées et stables.
- [ ] Tests unitaires dédiés verts.

## Validation commands
- `poetry run pytest -q tests/market_intelligence`

## Notes / pitfalls
- Standardiser timezone en UTC.
- Politique explicite de forward-fill et trous de données.
