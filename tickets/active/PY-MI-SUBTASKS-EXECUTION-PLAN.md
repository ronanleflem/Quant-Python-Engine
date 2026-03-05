## PY-MI — Plan d’exécution détaillé (autonome pour agents Codex)

Objectif: rendre chaque sous-tâche exécutable sans ambiguïté par un agent.

---

## Mode opératoire standard (à appliquer à **chaque** sous-tâche)

### Pré-checks obligatoires
1. Lire l’epic cible (`PY-MI-EPIC-x`).
2. Vérifier l’état git propre.
3. Identifier exactement les fichiers autorisés à modifier.
4. Lister les tests minimaux à exécuter avant PR.

### Convention branch / commit / PR
- Branch: `codex/py-mi-<subtask-id>-<slug>`
- Commit: `PY-MI-<id>: <résumé court>`
- PR title: `PY-MI-<id> <résumé>`
- PR body doit inclure:
  - Scope exact
  - Risques
  - Tests exécutés + résultat
  - Rollback plan

### Done minimal (pour chaque sous-tâche)
- [ ] Code + tests + doc locale mis à jour.
- [ ] Pas de rupture API involontaire.
- [ ] `pytest` ciblé vert.
- [ ] Diff borné au scope.

---

## EPIC-1 — Core contracts + import guards

### PY-MI-1.1 — `core/domain/models.py`
**Mission**
- Introduire modèles de domaine immuables: `Candle`, `Trade`, `Position`, `Portfolio`.

**Fichiers cibles**
- `src/quant_engine/core/domain/models.py` (nouveau)
- `src/quant_engine/core/domain/__init__.py` (nouveau)
- `tests/core/test_domain_models.py` (nouveau)

**Étapes d’implémentation**
1. Créer dataclasses `frozen=True` + types explicites.
2. Ajouter méthodes utilitaires minimales (`to_dict` optionnel, sans dépendance externe).
3. Exposer imports via `__init__.py`.
4. Écrire tests:
   - création objet,
   - immutabilité,
   - champs obligatoires,
   - comparaison/égalité.

**Tests à exécuter**
- `poetry run pytest -q tests/core/test_domain_models.py`

**Critères d’acceptation**
- Les 4 modèles sont importables depuis `quant_engine.core.domain`.
- Les tests d’immuabilité passent.

---

### PY-MI-1.2 — `core/contracts/*` (Protocols)
**Mission**
- Formaliser contrats `MarketIntelligenceService`, `FeatureStore`, `StrategyContract`.

**Fichiers cibles**
- `src/quant_engine/core/contracts/market_intelligence.py`
- `src/quant_engine/core/contracts/feature_store.py`
- `src/quant_engine/core/contracts/strategy.py`
- `src/quant_engine/core/contracts/__init__.py`
- `tests/core/test_contracts_imports.py`

**Étapes d’implémentation**
1. Définir `Protocol` + signatures minimales.
2. Utiliser types pandas seulement si nécessaire (`typing.TYPE_CHECKING`).
3. Ajouter docstrings de contrat (inputs/outputs attendus).
4. Tester import + conformité d’une implémentation fake.

**Tests à exécuter**
- `poetry run pytest -q tests/core/test_contracts_imports.py`

**Critères d’acceptation**
- Contrats importables et utilisables par classes fake.

---

### PY-MI-1.3 — Architecture import guards
**Mission**
- Empêcher les dépendances interdites via tests d’architecture.

**Fichiers cibles**
- `tests/architecture/test_import_rules.py`

**Étapes d’implémentation**
1. Scanner les imports AST des modules `src/quant_engine`.
2. Définir matrice `FORBIDDEN_IMPORTS` (ex: `core` -> pas `api/backtest/optimize`).
3. Faire échouer avec message clair indiquant source/target interdits.

**Tests à exécuter**
- `poetry run pytest -q tests/architecture/test_import_rules.py`

**Critères d’acceptation**
- Test échoue en cas de violation reproduite localement.

---

### PY-MI-1.4 — Adapters legacy minimaux
**Mission**
- Introduire adaptation de l’existant vers les nouveaux contrats sans break.

**Fichiers cibles**
- `src/quant_engine/market_intelligence/adapters/legacy_filters_adapter.py` (nouveau)
- `src/quant_engine/market_intelligence/adapters/legacy_stats_adapter.py` (nouveau)
- `tests/market_intelligence/test_legacy_adapters.py`

**Étapes d’implémentation**
1. Créer wrappers pour appeler `filters` / `stats` existants.
2. Uniformiser retour (`DataFrame` indexé UTC + colonnes normalisées).
3. Ajouter tests sur shape et colonnes.

**Tests à exécuter**
- `poetry run pytest -q tests/market_intelligence/test_legacy_adapters.py`

**Critères d’acceptation**
- Aucune régression visible sur flows existants.

---

## EPIC-2 — Feature Store + MI Service

### PY-MI-2.1 — Modèles MI
**Mission**
- Définir structures `FeatureMeta`, conventions de colonnes et labels.

**Fichiers cibles**
- `src/quant_engine/market_intelligence/models.py`
- `tests/market_intelligence/test_models.py`

**Étapes**
1. Définir schéma minimal des métadonnées.
2. Ajouter validateurs simples (version, timeframe, symbol).
3. Tester validation positive/négative.

**Tests**
- `poetry run pytest -q tests/market_intelligence/test_models.py`

---

### PY-MI-2.2 — InMemoryFeatureStore
**Mission**
- Stockage en RAM par clé `(feature_set, symbol, timeframe, version)`.

**Fichiers cibles**
- `src/quant_engine/market_intelligence/feature_store_memory.py`
- `tests/market_intelligence/test_feature_store_memory.py`

**Étapes**
1. Implémenter `get/put/exists/delete`.
2. Gérer overwrite explicite.
3. Tests concurrence simple (écritures séquentielles).

**Tests**
- `poetry run pytest -q tests/market_intelligence/test_feature_store_memory.py`

---

### PY-MI-2.3 — ParquetFeatureStore
**Mission**
- Persistance disque partitionnée et versionnée.

**Fichiers cibles**
- `src/quant_engine/market_intelligence/feature_store_parquet.py`
- `tests/market_intelligence/test_feature_store_parquet.py`

**Étapes**
1. Définir layout de path stable.
2. Implémenter write atomique (tmp + rename).
3. Implémenter lecture filtrée période.
4. Tests round-trip + version mismatch.

**Tests**
- `poetry run pytest -q tests/market_intelligence/test_feature_store_parquet.py`

---

### PY-MI-2.4 — MarketIntelligenceService v1
**Mission**
- Service unique pour calcul features corrélation/régime/liquidité.

**Fichiers cibles**
- `src/quant_engine/market_intelligence/service.py`
- `src/quant_engine/market_intelligence/pipeline.py`
- `tests/market_intelligence/test_service_v1.py`

**Étapes**
1. `compute_features(...)` + `label_regimes(...)` + `liquidity_flags(...)`.
2. Brancher adapters legacy en interne (phase transitoire).
3. Ajouter cache store read-through/write-through.
4. Tester déterminisme sur dataset fixe.

**Tests**
- `poetry run pytest -q tests/market_intelligence/test_service_v1.py`

---

### PY-MI-2.5 — Qualité de données features
**Mission**
- Encadrer index UTC, NaN policy, colonnes obligatoires.

**Fichiers cibles**
- `tests/market_intelligence/test_feature_quality_contract.py`

**Étapes**
1. Créer assertions réutilisables (`assert_feature_frame_contract`).
2. Tester monotonicité index, timezone UTC, colonnes minimales.
3. Tester policy NaN par feature.

**Tests**
- `poetry run pytest -q tests/market_intelligence/test_feature_quality_contract.py`

---

## EPIC-3 — Intégration moteurs

### PY-MI-3.1 — Injection MI dans backtest
**Mission**
- Permettre au runner backtest de consommer `MarketIntelligenceService`.

**Fichiers cibles**
- `src/quant_engine/backtest/runner.py`
- `tests/backtest/test_backtest_with_mi_injection.py`

**Étapes**
1. Ajouter param optionnel `mi_service` + fallback null adapter.
2. Résoudre features avant boucle d’exécution.
3. Passer features au composant de décision.

**Tests**
- `poetry run pytest -q tests/backtest/test_backtest_with_mi_injection.py`

---

### PY-MI-3.2 — Injection MI dans stratégie/DCA
**Mission**
- Faire consommer `features_row` par stratégies, y compris DCA.

**Fichiers cibles**
- `src/quant_engine/strategies/base.py`
- `src/quant_engine/strategies/runner.py`
- `tests/strategies/test_strategy_consumes_features.py`

**Étapes**
1. Étendre contrat stratégie (`on_bar(..., features_row, ...)`).
2. Adapter runner pour extraire row features par timestamp.
3. Maintenir compat backward (`features_row=None`).

**Tests**
- `poetry run pytest -q tests/strategies/test_strategy_consumes_features.py`

---

### PY-MI-3.3 — Mode fallback MI OFF
**Mission**
- Activer/désactiver MI par config sans casser les runs.

**Fichiers cibles**
- `src/quant_engine/config.py`
- `src/quant_engine/backtest/runner.py`
- `src/quant_engine/strategies/runner.py`
- `tests/integration/test_mi_toggle_on_off.py`

**Étapes**
1. Ajouter flag config `market_intelligence.enabled`.
2. En OFF: comportement identique legacy.
3. En ON: service MI utilisé.

**Tests**
- `poetry run pytest -q tests/integration/test_mi_toggle_on_off.py`

---

### PY-MI-3.4 — Stress tests sensibles au régime
**Mission**
- Ajouter chocs dépendants des labels MI (regime shift / vol spike).

**Fichiers cibles**
- `src/quant_engine/performance/stress_tests.py`
- `tests/integration/test_stress_regime_shift.py`

**Étapes**
1. Ajouter paramètre `regime_labels` au pipeline stress.
2. Moduler scénarios selon régime.
3. Tracer paramètres appliqués dans payload.

**Tests**
- `poetry run pytest -q tests/integration/test_stress_regime_shift.py`

---

### PY-MI-3.5 — Intégration ON/OFF + snapshots
**Mission**
- Garantir forme stable des payloads en mode MI ON/OFF.

**Fichiers cibles**
- `tests/integration/test_payload_snapshot_mi.py`
- `tests/data/golden/*` (si nécessaire)

**Étapes**
1. Générer snapshots de référence.
2. Comparer champs invariants + champs MI additionnels.
3. Documenter exceptions tolérées.

**Tests**
- `poetry run pytest -q tests/integration/test_payload_snapshot_mi.py`

---

## EPIC-4 — Analytics + stats/seasonality/optimize

### PY-MI-4.1 — Segmentation performance
**Mission**
- Exposer métriques segmentées par `regime` et `magnet_failure`.

**Fichiers cibles**
- `src/quant_engine/performance/backtest_builder.py`
- `src/quant_engine/performance/dca_builder.py`
- `tests/performance/test_segmentation_by_regime.py`

**Étapes**
1. Ajouter agrégations conditionnelles.
2. Intégrer au payload sans casser format existant.
3. Tester sur échantillons multi-régimes.

**Tests**
- `poetry run pytest -q tests/performance/test_segmentation_by_regime.py`

---

### PY-MI-4.2 — Migration partielle stats -> MI
**Mission**
- Déplacer calculs feature-like depuis `stats` vers MI avec wrappers compat.

**Fichiers cibles**
- `src/quant_engine/stats/events.py`
- `src/quant_engine/stats/conditions.py`
- `src/quant_engine/market_intelligence/` (nouveaux composants)
- `tests/stats/test_stats_compat_wrappers.py`

**Étapes**
1. Identifier fonctions purement feature.
2. Déplacer vers MI.
3. Laisser wrappers dans `stats` (même signature).
4. Marquer wrappers dépréciés (warning soft).

**Tests**
- `poetry run pytest -q tests/stats/test_stats_compat_wrappers.py`

---

### PY-MI-4.3 — Seasonality segmentée par labels MI
**Mission**
- Rendre la segmentation seasonality optionnelle via labels MI.

**Fichiers cibles**
- `src/quant_engine/seasonality/compute.py`
- `src/quant_engine/seasonality/runner.py`
- `tests/integration/test_seasonality_with_mi_labels.py`

**Étapes**
1. Ajouter option `segment_by_mi_labels`.
2. Joindre labels MI sur index temps.
3. Sortir stats globales + segmentées.

**Tests**
- `poetry run pytest -q tests/integration/test_seasonality_with_mi_labels.py`

---

### PY-MI-4.4 — Alignement optimize backtest + DCA
**Mission**
- Standardiser consommation des contrats MI dans flux optimize backtest/DCA.

**Fichiers cibles**
- `src/quant_engine/optimize/variants.py`
- `src/quant_engine/api/app.py`
- `tests/integration/test_optimize_backtest_dca_mi_alignment.py`

**Étapes**
1. Vérifier injection cohérente MI dans deux branches optimize.
2. Uniformiser shape des metadata MI dans résultats.
3. Tester `optimize_backtest` et `optimize_dca`.

**Tests**
- `poetry run pytest -q tests/integration/test_optimize_backtest_dca_mi_alignment.py`

---

### PY-MI-4.5 — Snapshot analytics globaux
**Mission**
- Stabiliser payload analytics/stats/seasonality en snapshots.

**Fichiers cibles**
- `tests/integration/test_analytics_snapshots.py`
- `tests/data/golden/*`

**Étapes**
1. Produire snapshots versionnés.
2. Ajouter stratégie d’update contrôlée.
3. Documenter champs volatils ignorés.

**Tests**
- `poetry run pytest -q tests/integration/test_analytics_snapshots.py`

---

## EPIC-5 — Migration/hardening final

### PY-MI-5.1 — Dépréciations progressives
**Mission**
- Introduire warnings de migration sans rupture.

**Fichiers cibles**
- `src/quant_engine/*` modules wrappers legacy
- `docs/` notes migration
- `tests/integration/test_deprecation_paths.py`

**Étapes**
1. Ajouter warnings explicites avec date/version cible.
2. Ajouter doc de migration old->new.
3. Tester présence warnings.

**Tests**
- `poetry run pytest -q tests/integration/test_deprecation_paths.py`

---

### PY-MI-5.2 — Découpage progressif `api/app.py`
**Mission**
- Extraire services applicatifs pour réduire couplage.

**Fichiers cibles**
- `src/quant_engine/api/app.py`
- `src/quant_engine/api/services/*.py` (nouveau)
- `tests/api/*`

**Étapes**
1. Extraire logique métier hors endpoints.
2. Garder routes minces (validation + appel service).
3. Assurer compat payload/status codes.

**Tests**
- `poetry run pytest -q tests/api`

---

### PY-MI-5.3 — Baseline KPI non-régression
**Mission**
- Comparer runs de référence avant/après.

**Fichiers cibles**
- `tests/integration/test_kpi_non_regression_mi.py`
- `tests/data/golden/*`

**Étapes**
1. Définir 2-3 specs de référence (backtest, dca, stress).
2. Comparer KPI clés avec tolérance.
3. Reporter diff en cas d’écart.

**Tests**
- `poetry run pytest -q tests/integration/test_kpi_non_regression_mi.py`

---

### PY-MI-5.4 — Renforcer architecture tests
**Mission**
- Élargir les règles d’import interdites à tous modules critiques.

**Fichiers cibles**
- `tests/architecture/test_import_rules.py`

**Étapes**
1. Ajouter toutes règles officielles de dépendance.
2. Tester cas négatifs synthétiques.
3. Ajouter message de remédiation.

**Tests**
- `poetry run pytest -q tests/architecture/test_import_rules.py`

---

### PY-MI-5.5 — Runbook opératoire final
**Mission**
- Documenter opérationnellement recalcul/inspection features et maintenance cache.

**Fichiers cibles**
- `docs/market_intelligence_runbook.md` (nouveau)
- `docs/architecture_market_intelligence_refactor.md` (liens)

**Étapes**
1. Documenter commandes CLI (`features recompute/inspect`).
2. Documenter invalidation cache et troubleshooting.
3. Ajouter checklist incident.

**Tests / checks**
- Vérification manuelle cohérence doc + commandes existantes.

---

## Checklists PR (copier-coller pour agents)

### Template "PR Summary"
- Scope:
- Fichiers modifiés:
- Compatibilité:
- Risques:
- Rollback:

### Template "Validation"
- ✅ `poetry run pytest -q <tests ciblés>`
- ✅ `poetry run pytest -q` (si coût acceptable)
- ✅ `poetry run ruff check src tests` (si configuré)

### Template "Reviewer notes"
- Contrats respectés ?
- Imports interdits respectés ?
- Données temporelles UTC garanties ?
- Diff borné au ticket ?
