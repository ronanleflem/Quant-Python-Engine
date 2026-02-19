## BMAD Stage
- PM

## Cross-Repo Coordination
- Cross-Repo Initiative: N/A
- Upstream Dependencies: None
- Contract Version: N/A

## Context7 Decision
- Required: No
- Reason: Audit and optimization scope uses local codebase and known tooling.

# Title
Audit et optimisation des variations de calcul (stratégies / backtests)

## Goal
Produire un inventaire clair des chemins de calcul existants (stratégies, backtests, optimisations), identifier des pistes d’optimisation concrètes et mettre en œuvre au moins une amélioration sûre et mesurable, sans changement de comportement externe.

## Context / Entry points
- `src/quant_engine/optimize/variants.py` (orchestration des variantes)
- `src/quant_engine/backtest/runner.py` (pipeline backtest)
- `src/quant_engine/strategies/runner.py` (pipeline stratégie)
- `src/quant_engine/performance/backtest_builder.py` (métriques backtest)
- `src/quant_engine/performance/dca_builder.py` (métriques DCA)
- `docs/optimization.md` (workflow attendu)
- `docs/tests_plan.md` (groupes de tests)
- `README.md` (commandes backtest/optimize/tests)

## Constraints & conventions
- Aucun changement de comportement externe (API, schémas, métriques, sorties).
- Pas de refactor hors scope : optimisations ciblées uniquement.
- Préserver le déterminisme et la reproductibilité.
- Préférer pandas/numpy vectorisé; éviter les boucles Python nouvelles.
- Respecter les conventions de nommage et patterns existants.

## Definition of Done (DoD)
- [ ] Inventaire des variantes de calcul et points d’entrée rédigé (paths + bref résumé).
- [ ] Au moins 3 pistes d’optimisation identifiées avec impact attendu + risques.
- [ ] Une optimisation low-risk implémentée et validée sans changement d’output.
- [ ] Tests unitaires/intégration pertinents passés.
- [ ] Sanity check perf exécuté (baseline vs after).
- [ ] Commandes de validation exécutées avec succès.

## Implementation plan
1. Cartographier les chemins de calcul existants :
   - stratégie backtest
   - backtest classique
   - optimisation (variants)
2. Documenter les redondances (features, métriques, conversions).
3. Proposer des optimisations candidates (cache, évitement de recompute, moins de conversions).
4. Sélectionner une optimisation low-risk et l’implémenter.
5. Mettre à jour la doc si une hypothèse est clarifiée.
6. Exécuter tests + perf sanity et comparer avant/après.

## Tests
- Unit: tests ciblés sur la zone modifiée.
- Integration: tests backtest/strategy/optimize existants liés au flux touché.
- Performance sanity: test large dataset existant.

## Fast tests vs Slow tests
- Fast: tests unitaires/intégration ciblés (ex. `tests/test_optimize_variants_baseline.py`).
- Slow: tests perf marqués `-m slow` (ex. `tests/test_large_dataset_perf.py`).

## Validation commands
- `poetry run pytest -q tests/test_optimize_variants_baseline.py`
- `poetry run pytest -q tests/test_strategy_dca_variants.py`
- `poetry run pytest -m slow -q tests/test_large_dataset_perf.py`

## Non-goals / Out of scope
- Pas de nouveaux algos d’optimisation.
- Pas de changements de schéma JSON / API.
- Pas de refactor global du moteur.
- Pas de modifications des connecteurs data.

## Notes / pitfalls
- Vérifier l’égalité stricte des outputs avant/après (métriques, trades, signaux).
- Attention aux différences numériques si l’ordre de calcul change.
- Documenter clairement toute hypothèse sur la forme des datasets.
