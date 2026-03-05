## Title
- PY-MI — Tickets prompts Codex exécutables (suite audit EPIC 1→5)

## Objectif
- Transformer le rapport d’audit PY-MI en tickets d’exécution directement utilisables par des agents Codex.
- Chaque ticket ci-dessous inclut: scope borné, étapes, commandes de validation et critères d’acceptation.

---

## Ticket 1 — PY-MI-5.6 Full regression gate CI (P0)

### Prompt agent (copier/coller)
Tu implémentes **PY-MI-5.6**.

Contexte:
- Le DoD de migration/hardening exige une validation globale `poetry run pytest -q`.
- On veut un gate CI bloquant avec artefacts de résultats.

Mission:
1. Ajouter/mettre à jour le workflow CI pour exécuter `poetry run pytest -q`.
2. Publier les artefacts (junit/log) en cas de succès/échec.
3. Échouer explicitement le workflow si les tests échouent.
4. Documenter la commande de reproduction locale dans `README` ou `docs/`.

Fichiers cibles:
- `.github/workflows/*`
- `docs/*` ou `README*`

Validation:
- `poetry run pytest -q`
- (si présent) `poetry run ruff check src tests`

Critères d’acceptation:
- Le workflow lance bien la suite complète.
- Le status PR est bloquant en cas d’échec.
- La doc indique comment reproduire localement.

---

## Ticket 2 — PY-MI-4.6 Optimize MI E2E hardening (P0)

### Prompt agent (copier/coller)
Tu implémentes **PY-MI-4.6**.

Contexte:
- Le test actuel d’alignement optimize MI valide surtout le contrat API (monkeypatch), pas un run E2E complet.

Mission:
1. Ajouter un test d’intégration E2E optimize backtest + optimize DCA **sans monkeypatch des runners**.
2. Vérifier que les metadata MI (`enabled`, `contract`) sont cohérentes dans les deux branches.
3. Vérifier la stabilité du shape résultat (champs invariants + champs MI).

Fichiers cibles:
- `tests/integration/test_optimize_backtest_dca_mi_e2e.py` (nouveau)
- `tests/data/golden/*` (si snapshot nécessaire)

Validation:
- `poetry run pytest -q tests/integration/test_optimize_backtest_dca_mi_alignment.py tests/integration/test_optimize_backtest_dca_mi_e2e.py`

Critères d’acceptation:
- Les deux flux optimize exécutent réellement leur chemin nominal.
- Le contrat MI est identique côté backtest et DCA.
- Aucun breaking change API.

---

## Ticket 3 — PY-MI-3.6 MI toggle consistency matrix (P1)

### Prompt agent (copier/coller)
Tu implémentes **PY-MI-3.6**.

Contexte:
- L’activation MI dépend de `env + settings + spec`; il faut éviter les divergences de comportement.

Mission:
1. Centraliser la résolution du toggle MI dans un helper unique partagé.
2. Couvrir les cas: env absent, env=true/false, override spec true/false, conflit env/spec.
3. Appliquer ce helper dans backtest + strategies.

Fichiers cibles:
- `src/quant_engine/backtest/runner.py`
- `src/quant_engine/strategies/runner.py`
- `src/quant_engine/config.py` (si nécessaire)
- `tests/integration/test_mi_toggle_consistency_matrix.py` (nouveau)

Validation:
- `poetry run pytest -q tests/integration/test_mi_toggle_on_off.py tests/integration/test_mi_toggle_consistency_matrix.py`

Critères d’acceptation:
- Une seule source de vérité de résolution du toggle.
- Matrice de cas couverte et verte.
- Compatibilité legacy conservée.

---

## Ticket 4 — PY-MI-1.5 Contract signature convergence (P1)

### Prompt agent (copier/coller)
Tu implémentes **PY-MI-1.5**.

Contexte:
- Les contrats core existent, mais la convergence exacte avec la cible architecture doit être clarifiée et figée.

Mission:
1. Comparer les signatures Protocol actuelles vs contrat cible d’architecture.
2. Soit aligner les signatures, soit documenter explicitement la divergence (ADR courte) + plan de migration.
3. Ajouter tests d’import/conformité mis à jour.

Fichiers cibles:
- `src/quant_engine/core/contracts/*.py`
- `tests/core/test_contracts_imports.py`
- `docs/architecture_market_intelligence_refactor.md` (si ajustement)

Validation:
- `poetry run pytest -q tests/core/test_contracts_imports.py tests/architecture/test_import_rules.py`

Critères d’acceptation:
- Contrats explicitement figés (code + doc).
- Aucune ambiguïté pour les implémenteurs.

---

## Ticket 5 — PY-MI-5.7 Deprecation cleanup wave-2 (P2)

### Prompt agent (copier/coller)
Tu implémentes **PY-MI-5.7**.

Contexte:
- La première vague de dépréciation est en place; il faut terminer l’inventaire et normaliser les warnings.

Mission:
1. Inventorier tous wrappers legacy restants côté MI/stats.
2. Uniformiser message de dépréciation (version cible + date + alternative).
3. Étendre les tests warnings pour couvrir les chemins manquants.

Fichiers cibles:
- `src/quant_engine/stats/*.py`
- `src/quant_engine/market_intelligence/adapters/*.py`
- `tests/integration/test_deprecation_paths.py`

Validation:
- `poetry run pytest -q tests/integration/test_deprecation_paths.py tests/stats/test_stats_compat_wrappers.py`

Critères d’acceptation:
- Tous les chemins legacy auditables émettent un warning homogène.
- La doc migration est alignée.

---

## Ticket 6 — PY-MI-4.7 Benchmark perf MI multi-symboles (P2)

### Prompt agent (copier/coller)
Tu implémentes **PY-MI-4.7**.

Contexte:
- Les comportements fonctionnels sont couverts; la capacité/performance MI multi-symboles doit être quantifiée.

Mission:
1. Ajouter un benchmark reproductible (script ou test perf non bloquant CI) pour MI ON/OFF.
2. Mesurer latence, mémoire, et overhead par symbole/timeframe.
3. Produire un rapport court dans `docs/` avec seuils/recommandations.

Fichiers cibles:
- `tests/perf/*` ou `scripts/*`
- `docs/*`

Validation:
- `poetry run pytest -q tests/perf -k mi` (ou commande script documentée)

Critères d’acceptation:
- Métriques comparables run-to-run.
- Résultat exploitable pour capacity planning.

---

## Ordre de livraison recommandé
1. PY-MI-5.6 (P0)
2. PY-MI-4.6 (P0)
3. PY-MI-3.6 (P1)
4. PY-MI-1.5 (P1)
5. PY-MI-5.7 (P2)
6. PY-MI-4.7 (P2)

