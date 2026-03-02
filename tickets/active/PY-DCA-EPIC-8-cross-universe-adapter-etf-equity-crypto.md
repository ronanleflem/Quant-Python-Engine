## Title
- PY-DCA-EPIC-8 — Mutualisation multi-univers via AssetUniverseAdapter (ETF/Equity/Crypto)

## Ticket type
- Type B: Implementation

## BMAD Stage
- Architect

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-DCA-GRID-001
- Repo Owner: quant-python-engine
- Upstream Dependencies: PY-DCA-EPIC-1, PY-DCA-EPIC-2, PY-DCA-EPIC-6
- Contract Version: dca-grid-process-v1

## Goal
- Mutualiser le moteur DCA et les métriques pour plusieurs univers d’actifs via un adapter dédié, sans dupliquer la logique core.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/strategies/runner.py`
  - `src/quant_engine/datafeeds/`
  - `src/quant_engine/backtest/metrics.py`
  - `specs/examples/strategy_dca_etf_delta_2024_2026.json`

## BMAD Handover In
- Contrat de données versionné EPIC-6.

## BMAD Handover Out
- Adapter univers prêt pour intégration UI et analyses comparatives.

## Context7 Decision
- Required: No
- Reason: Architecture interne existante suffisante.

## Definition of Done
- [ ] Interface `AssetUniverseAdapter` implémentée + implémentations ETF/Equity/Crypto.
- [ ] Paramètres univers (calendrier, lot/fraction, frais, corporate actions) externalisés.
- [ ] Règle `universe_rules_version` persistée dans les artefacts.
- [ ] Tests cross-universe de non-régression.

## Implementation plan
1. Définir interface adapter et points d’injection dans runner DCA.
2. Implémenter adaptateurs concrets + mappings config/spec.
3. Ajouter tests et exemples specs par univers.

## Tests
- Unit tests:
  - `tests/strategies/test_asset_universe_adapter.py`
- Integration tests:
  - `tests/integration/test_dca_cross_universe_specs.py`

## Validation commands
- `poetry run pytest -q tests/strategies/test_asset_universe_adapter.py`
- `poetry run pytest -q tests/integration/test_dca_cross_universe_specs.py`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Comparaison de performance inter-univers sans normalisation risque/frais/calendrier.

## Notes / pitfalls
- Crypto 24/7 impose des règles calendaires incompatibles avec turn-of-month “jours ouvrés” standards.
