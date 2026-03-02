# PY-DCA — Plan de sous-tickets détaillé (actionnable par agents Codex)

Objectif: fournir des sous-tickets suffisamment précis pour qu’un agent exécute sans ambiguïté (scope, fichiers, tests, commandes, dépendances, risques).

## 1) Règles opératoires (obligatoires)
- 1 PR = 1 sous-ticket.
- Taille cible: 1 à 2 jours agent.
- Toute PR doit inclure: implémentation, tests, mise à jour doc, et validation locale.
- Tout changement de contrat (payload/artefact/API) doit être versionné et documenté.
- Interdiction de “batcher” plusieurs sous-tickets sauf mention explicite dans ce plan.

## 2) Template d’exécution d’un sous-ticket
Pour chaque sous-ticket, l’agent doit renseigner dans la PR:
1. **Scope précis** (ce qui est inclus / exclu).
2. **Fichiers touchés** (liste explicite).
3. **Tests ajoutés/modifiés** (unit + integration).
4. **Commandes de validation** exécutées avec résultats.
5. **Risques** et mesures de mitigation.

---

## 3) Sous-tickets détaillés

## EPIC-1 — Benchmarks calendaires & runners passifs

### PY-DCA-1.1 — Contrat benchmark DCA
- Scope: interface commune des variantes passives + schéma de sortie unique.
- Fichiers cibles:
  - `src/quant_engine/strategies/runner.py`
  - `docs/dca_grid_implementation_process.md`
- Tests:
  - `tests/strategies/test_dca_benchmark_calendars.py`
- Validation:
  - `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py`
- Dépendances: aucune.
- Risque: divergence de schéma entre variantes.

### PY-DCA-1.2 — Mensuel fixe
- Scope: implémentation calendrier jour fixe + fallback non ouvré documenté.
- Fichiers cibles:
  - `src/quant_engine/strategies/runner.py`
  - `specs/tests/strategy_dca_equity_monthly_fixed.json`
- Tests:
  - `tests/strategies/test_dca_benchmark_calendars.py`
- Validation:
  - `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py`

### PY-DCA-1.3 — Mensuel randomisé
- Scope: randomisation calendrier contrôlée par seed.
- Fichiers cibles:
  - `src/quant_engine/strategies/runner.py`
  - `specs/tests/strategy_dca_equity_monthly_randomized.json`
- Tests:
  - `tests/strategies/test_dca_benchmark_calendars.py`
- Validation:
  - `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py`
- Risque: non-déterminisme si seed non propagée.

### PY-DCA-1.4 — Mid-month + Turn-of-month
- Scope: implémenter les 2 variantes + cas limites (mois courts/week-end).
- Fichiers cibles:
  - `src/quant_engine/strategies/runner.py`
  - `specs/tests/strategy_dca_equity_mid_month.json`
  - `specs/tests/strategy_dca_equity_turn_of_month.json`
- Tests:
  - `tests/strategies/test_dca_benchmark_calendars.py`
- Validation:
  - `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py`

### PY-DCA-1.5 — Hebdo fixe
- Scope: exécution hebdo + contrôle timezone.
- Fichiers cibles:
  - `src/quant_engine/strategies/runner.py`
- Tests:
  - `tests/strategies/test_dca_benchmark_calendars.py`
- Validation:
  - `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py`

### PY-DCA-1.6 — Intégration CLI/API
- Scope: exposer variantes passives dans pipeline run-local/submit.
- Fichiers cibles:
  - `src/quant_engine/cli/main.py`
  - `src/quant_engine/api/app.py`
  - `specs/tests/strategy_dca_equity_benchmark_matrix.json`
- Tests:
  - `tests/integration/test_dca_benchmark_specs.py`
- Validation:
  - `poetry run pytest -q tests/integration/test_dca_benchmark_specs.py`

---

## EPIC-2 — Moteur de métriques

### PY-DCA-2.1 — final_performance_normalized
- Fichiers cibles: `src/quant_engine/backtest/metrics.py`
- Tests: `tests/backtest/test_dca_metrics.py`
- Validation: `poetry run pytest -q tests/backtest/test_dca_metrics.py`

### PY-DCA-2.2 — TWR
- Fichiers cibles: `src/quant_engine/backtest/metrics.py`
- Tests: `tests/backtest/test_dca_metrics.py`
- Validation: `poetry run pytest -q tests/backtest/test_dca_metrics.py`

### PY-DCA-2.3 — XIRR robuste
- Fichiers cibles: `src/quant_engine/backtest/metrics.py`
- Tests: `tests/backtest/test_dca_metrics.py`
- Validation: `poetry run pytest -q tests/backtest/test_dca_metrics.py`
- Risque: non-convergence -> statut explicite requis, pas de fallback silencieux.

### PY-DCA-2.4 — Max drawdown sur capital contribué
- Fichiers cibles: `src/quant_engine/backtest/metrics.py`
- Tests: `tests/backtest/test_dca_metrics.py`
- Validation: `poetry run pytest -q tests/backtest/test_dca_metrics.py`

### PY-DCA-2.5 — Time under water
- Fichiers cibles: `src/quant_engine/backtest/metrics.py`
- Tests: `tests/backtest/test_dca_metrics.py`
- Validation: `poetry run pytest -q tests/backtest/test_dca_metrics.py`

### PY-DCA-2.6 — Intégration persistence + API
- Fichiers cibles:
  - `src/quant_engine/persistence/db.py`
  - `src/quant_engine/api/app.py`
- Tests:
  - `tests/api/test_runs_metrics_dca.py`
- Validation:
  - `poetry run pytest -q tests/api/test_runs_metrics_dca.py`

---

## EPIC-3 — Robustesse statistique

### PY-DCA-3.1 — Job Monte Carlo calendrier
- Fichiers cibles: `src/quant_engine/stats/runner.py`
- Tests: `tests/stats/test_dca_robustness.py`
- Validation: `poetry run pytest -q tests/stats/test_dca_robustness.py`

### PY-DCA-3.2 — Agrégats distribution
- Fichiers cibles: `src/quant_engine/stats/estimators.py`
- Tests: `tests/stats/test_dca_robustness.py`
- Validation: `poetry run pytest -q tests/stats/test_dca_robustness.py`

### PY-DCA-3.3 — Percentile grid-vs-passive
- Fichiers cibles:
  - `src/quant_engine/stats/runner.py`
  - `src/quant_engine/persistence/db.py`
- Tests: `tests/stats/test_dca_robustness.py`
- Validation: `poetry run pytest -q tests/stats/test_dca_robustness.py`

### PY-DCA-3.4 — Dominance simplifiée
- Fichiers cibles:
  - `src/quant_engine/stats/estimators.py`
  - `docs/dca_grid_implementation_process.md`
- Tests: `tests/stats/test_dca_robustness.py`
- Validation: `poetry run pytest -q tests/stats/test_dca_robustness.py`

### PY-DCA-3.5 — Stress test paramètres grille
- Fichiers cibles: `src/quant_engine/optimize/runner.py`
- Tests: `tests/test_dca_stress_tests.py`
- Validation: `poetry run pytest -q tests/test_dca_stress_tests.py`

---

## EPIC-4 — Rolling windows & régimes
- **PY-DCA-4.1** Splitter rolling (`src/quant_engine/validate/splitter.py`) + `tests/validate/test_dca_rolling_windows.py`
- **PY-DCA-4.2** Métriques par fenêtre (`src/quant_engine/strategies/runner.py`) + `tests/integration/test_dca_rolling_regimes_pipeline.py`
- **PY-DCA-4.3** IRR glissant (`src/quant_engine/backtest/metrics.py`) + `tests/integration/test_dca_rolling_regimes_pipeline.py`
- **PY-DCA-4.4** Sous-performance structurelle (`src/quant_engine/stats/runner.py`) + `tests/integration/test_dca_rolling_regimes_pipeline.py`
- Validation commune:
  - `poetry run pytest -q tests/validate/test_dca_rolling_windows.py tests/integration/test_dca_rolling_regimes_pipeline.py`

## EPIC-5 — Score composite
- **PY-DCA-5.1** Contrat score + config poids (`src/quant_engine/backtest/metrics.py`, docs)
- **PY-DCA-5.2** Calcul score + persistance (`src/quant_engine/persistence/db.py`)
- **PY-DCA-5.3** Edge buckets faible/moyen/fort (`src/quant_engine/api/app.py`)
- **PY-DCA-5.4** Sensibilité poids (`tests/integration/test_dca_score_sensitivity.py`)
- Validation commune:
  - `poetry run pytest -q tests/backtest/test_dca_score.py tests/integration/test_dca_score_sensitivity.py`

## EPIC-6 — Contrat de données versionné
- **PY-DCA-6.1** Schémas artefacts + `schema_version` (`src/quant_engine/io/`)
- **PY-DCA-6.2** Écriture JSON/Parquet normalisée (`src/quant_engine/io/`)
- **PY-DCA-6.3** Endpoints artefacts run (`src/quant_engine/api/app.py`)
- **PY-DCA-6.4** Compatibilité ascendante (tests dédiés)
- Validation commune:
  - `poetry run pytest -q tests/io/test_dca_artifact_contract.py tests/api/test_dca_artifact_endpoints.py`

## EPIC-7 — Méthodologie & audit
- **PY-DCA-7.1** Checklist anti-snooping (`docs/`)
- **PY-DCA-7.2** Matrice risques + mitigation (`docs/`)
- **PY-DCA-7.3** Template audit trail (`docs/` + metadata run)
- **PY-DCA-7.4** Procédure rerun canonique (`docs/` + `specs/examples/`)
- Validation:
  - `poetry run qe run-local --spec specs/examples/strategy_dca_etf_delta_2024_2026.json`

## EPIC-8 — Multi-univers ETF/Equity/Crypto
- **PY-DCA-8.1** Interface `AssetUniverseAdapter` (injection runner)
- **PY-DCA-8.2** Adapter ETF
- **PY-DCA-8.3** Adapter Equity (corporate actions)
- **PY-DCA-8.4** Adapter Crypto (24/7)
- **PY-DCA-8.5** `universe_rules_version` dans artefacts
- **PY-DCA-8.6** Tests non-régression cross-universe
- Fichiers cibles:
  - `src/quant_engine/strategies/runner.py`
  - `src/quant_engine/datafeeds/`
  - `src/quant_engine/io/`
  - `specs/examples/`
- Validation:
  - `poetry run pytest -q tests/strategies/test_asset_universe_adapter.py tests/integration/test_dca_cross_universe_specs.py`

---

## Annexe A — Mapping officiel sous-ticket -> tests existants -> commande CI

> Objectif: garantir une traçabilité **exécutable telle quelle** sans renommer massivement les fichiers de tests.
> Règle: en cas d’écart entre les sections ci-dessus et ce tableau, **ce tableau fait foi pour l’audit CI**.

| Sous-ticket | Tests existants (repo) | Commande CI exécutable |
|---|---|---|
| PY-DCA-1.1 | `tests/strategies/test_dca_benchmark_calendars.py`, `tests/integration/test_dca_benchmark_specs.py` | `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py tests/integration/test_dca_benchmark_specs.py` |
| PY-DCA-1.2 | `tests/strategies/test_dca_benchmark_calendars.py` | `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py` |
| PY-DCA-1.3 | `tests/strategies/test_dca_benchmark_calendars.py` | `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py` |
| PY-DCA-1.4 | `tests/strategies/test_dca_benchmark_calendars.py` | `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py` |
| PY-DCA-1.5 | `tests/strategies/test_dca_benchmark_calendars.py` | `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py` |
| PY-DCA-1.6 | `tests/integration/test_dca_benchmark_specs.py`, `tests/test_cli_smoke.py` | `poetry run pytest -q tests/integration/test_dca_benchmark_specs.py tests/test_cli_smoke.py` |
| PY-DCA-2.1 | `tests/backtest/test_dca_metrics.py` | `poetry run pytest -q tests/backtest/test_dca_metrics.py` |
| PY-DCA-2.2 | `tests/backtest/test_dca_metrics.py` | `poetry run pytest -q tests/backtest/test_dca_metrics.py` |
| PY-DCA-2.3 | `tests/backtest/test_dca_metrics.py` | `poetry run pytest -q tests/backtest/test_dca_metrics.py` |
| PY-DCA-2.4 | `tests/backtest/test_dca_metrics.py` | `poetry run pytest -q tests/backtest/test_dca_metrics.py` |
| PY-DCA-2.5 | `tests/backtest/test_dca_metrics.py` | `poetry run pytest -q tests/backtest/test_dca_metrics.py` |
| PY-DCA-2.6 | `tests/api/test_runs_metrics_dca.py` | `poetry run pytest -q tests/api/test_runs_metrics_dca.py` |
| PY-DCA-3.1 | `tests/stats/test_dca_robustness.py`, `tests/integration/test_dca_robustness_pipeline.py` | `poetry run pytest -q tests/stats/test_dca_robustness.py tests/integration/test_dca_robustness_pipeline.py` |
| PY-DCA-3.2 | `tests/stats/test_dca_robustness.py` | `poetry run pytest -q tests/stats/test_dca_robustness.py` |
| PY-DCA-3.3 | `tests/stats/test_dca_robustness.py` | `poetry run pytest -q tests/stats/test_dca_robustness.py` |
| PY-DCA-3.4 | `tests/stats/test_dca_robustness.py` | `poetry run pytest -q tests/stats/test_dca_robustness.py` |
| PY-DCA-3.5 | `tests/test_dca_stress_tests.py` | `poetry run pytest -q tests/test_dca_stress_tests.py` |
| PY-DCA-4.1 | `tests/validate/test_dca_rolling_windows.py` | `poetry run pytest -q tests/validate/test_dca_rolling_windows.py` |
| PY-DCA-4.2 | `tests/integration/test_dca_rolling_regimes_pipeline.py` | `poetry run pytest -q tests/integration/test_dca_rolling_regimes_pipeline.py` |
| PY-DCA-4.3 | `tests/integration/test_dca_rolling_regimes_pipeline.py` | `poetry run pytest -q tests/integration/test_dca_rolling_regimes_pipeline.py` |
| PY-DCA-4.4 | `tests/integration/test_dca_rolling_regimes_pipeline.py` | `poetry run pytest -q tests/integration/test_dca_rolling_regimes_pipeline.py` |
| PY-DCA-5.1 | `tests/backtest/test_dca_score.py` | `poetry run pytest -q tests/backtest/test_dca_score.py` |
| PY-DCA-5.2 | `tests/backtest/test_dca_score.py`, `tests/test_persistence_db.py` | `poetry run pytest -q tests/backtest/test_dca_score.py tests/test_persistence_db.py` |
| PY-DCA-5.3 | `tests/test_api_schemas_contract.py`, `tests/test_api_endpoints_extended.py` | `poetry run pytest -q tests/test_api_schemas_contract.py tests/test_api_endpoints_extended.py` |
| PY-DCA-5.4 | `tests/integration/test_dca_score_sensitivity.py` | `poetry run pytest -q tests/integration/test_dca_score_sensitivity.py` |
| PY-DCA-6.1 | `tests/io/test_dca_artifact_contract.py` | `poetry run pytest -q tests/io/test_dca_artifact_contract.py` |
| PY-DCA-6.2 | `tests/io/test_dca_artifact_contract.py` | `poetry run pytest -q tests/io/test_dca_artifact_contract.py` |
| PY-DCA-6.3 | `tests/api/test_dca_artifact_endpoints.py` | `poetry run pytest -q tests/api/test_dca_artifact_endpoints.py` |
| PY-DCA-6.4 | `tests/io/test_dca_artifact_contract.py`, `tests/api/test_dca_artifact_endpoints.py` | `poetry run pytest -q tests/io/test_dca_artifact_contract.py tests/api/test_dca_artifact_endpoints.py` |
| PY-DCA-7.1 | `tests/test_backtest_engine_no_lookahead.py` | `poetry run pytest -q tests/test_backtest_engine_no_lookahead.py` |
| PY-DCA-7.2 | `tests/strategies/test_dca_benchmark_calendars.py` | `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py` |
| PY-DCA-7.3 | `tests/integration/test_dca_benchmark_specs.py` | `poetry run pytest -q tests/integration/test_dca_benchmark_specs.py` |
| PY-DCA-7.4 | `tests/integration/test_dca_robustness_pipeline.py` | `poetry run pytest -q tests/integration/test_dca_robustness_pipeline.py` |
| PY-DCA-8.1 | `tests/strategies/test_asset_universe_adapter.py` | `poetry run pytest -q tests/strategies/test_asset_universe_adapter.py` |
| PY-DCA-8.2 | `tests/integration/test_dca_cross_universe_specs.py` | `poetry run pytest -q tests/integration/test_dca_cross_universe_specs.py` |
| PY-DCA-8.3 | `tests/integration/test_dca_cross_universe_specs.py` | `poetry run pytest -q tests/integration/test_dca_cross_universe_specs.py` |
| PY-DCA-8.4 | `tests/integration/test_dca_cross_universe_specs.py` | `poetry run pytest -q tests/integration/test_dca_cross_universe_specs.py` |
| PY-DCA-8.5 | `tests/io/test_dca_artifact_contract.py`, `tests/integration/test_dca_cross_universe_specs.py` | `poetry run pytest -q tests/io/test_dca_artifact_contract.py tests/integration/test_dca_cross_universe_specs.py` |
| PY-DCA-8.6 | `tests/strategies/test_asset_universe_adapter.py`, `tests/integration/test_dca_cross_universe_specs.py` | `poetry run pytest -q tests/strategies/test_asset_universe_adapter.py tests/integration/test_dca_cross_universe_specs.py` |

---

## 4) Orchestration multi-agents (recommandée)
- Vague A (fondations): 1.1, 2.1, 6.1
- Vague B (implémentation core): 1.2→1.6, 2.2→2.6
- Vague C (analytics): 3.1→3.5, 4.1→4.4
- Vague D (décision): 5.1→5.4
- Vague E (multi-univers): 8.1→8.6
- Vague F (gouvernance): 7.1→7.4

## 5) Definition of Ready (DoR)
- Scope borné à 1 module principal + 1 module d’intégration max.
- Dépendances mergées confirmées.
- Jeu de données/spec de test identifié.
- Commandes de validation listées avant implémentation.

## 6) Definition of Done (DoD)
- [ ] Code implémenté selon scope.
- [ ] Tests unitaires + intégration pertinents.
- [ ] Commandes exécutées et résultats reportés dans la PR.
- [ ] Docs / tickets impactés mis à jour.
- [ ] Pas de changement de contrat non versionné.
