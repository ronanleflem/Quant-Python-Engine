# Process d’implémentation — DCA Grid vs Benchmarks passifs (Python puis Angular)

## 1) Audit du brouillon initial

Le brouillon est pertinent et couvre les bons axes (benchmarks, métriques, robustesse, reporting, méthodologie), mais il mélange :

- des **fonctionnalités de calcul** (moteur quant Python),
- des **analyses de recherche** (validation statistique),
- des **besoins produit/UI** (Angular, visualisation),
- et des **conclusions métier** (alpha vs confort psychologique).

Pour exécuter proprement, il faut découpler en deux streams :

1. **Stream Python (source de vérité)** : génération des résultats, métriques et score reproductibles.
2. **Stream Angular (consommation)** : visualisation, interprétation, export, pilotage.

---

## 2) Cadrage cible (définitions à geler avant développement)

### 2.1 Périmètre de benchmark (contrat ex-ante)

- Actif(s) éligible(s), fuseau horaire, calendrier de trading, période d’étude.
- Cashflow identique entre stratégies (capital investi égal à date égale ou budget strict équivalent).
- Frais, slippage, taxes, arrondis (unités fractionnaires autorisées ou non).
- Règle de disponibilité des prix (jour férié/week-end => décalage ou skip).

### 2.2 Variantes de stratégies passives à implémenter

- DCA mensuel naïf (jour fixe).
- DCA mensuel randomisé (Monte Carlo sur le calendrier d’exécution).
- DCA mid-month (jours 10–20).
- DCA turn-of-the-month (-3 / +3 jours ouvrés).
- DCA hebdomadaire passif (jour fixe).

### 2.3 Grille DCA (stratégie comparée)

- Paramètres de grille gelés ex-ante (pas d’optimisation opportuniste post-hoc).
- Versionning strict des paramètres pour reproductibilité.

---

## 3) Roadmap d’implémentation côté Python (priorité 1)

## Epic PY-1 — Moteur de benchmarks & calendrier

### Tickets

- **PY-1.1** Définir une interface commune `StrategyRunner` (inputs, outputs, metadata).
- **PY-1.2** Implémenter DCA mensuel naïf.
- **PY-1.3** Implémenter DCA mensuel randomisé (N simulations, seed fixe).
- **PY-1.4** Implémenter DCA mid-month.
- **PY-1.5** Implémenter DCA turn-of-the-month.
- **PY-1.6** Implémenter DCA hebdomadaire passif.
- **PY-1.7** Ajouter tests unitaires calendrier (jours non ouvrés, mois courts, fuseau).

### DoD

- Même schéma de sortie pour toutes les variantes.
- Reproductibilité garantie via seed + config immuable.
- Couverture de tests minimale sur cas limites calendaires.

---

## Epic PY-2 — Moteur de métriques financières

### Tickets

- **PY-2.1** Performance finale normalisée (capital investi égal).
- **PY-2.2** TWR (Time-Weighted Return).
- **PY-2.3** IRR/XIRR robuste (cashflows irréguliers).
- **PY-2.4** Max drawdown sur capital investi.
- **PY-2.5** Time to recovery / temps sous l’eau.
- **PY-2.6** Validations numériques (bornes, invariants, tests de non-régression).

### DoD

- API métriques stable (typing + docstring + exemples).
- Résultats cohérents sur datasets synthétiques de référence.
- Tolérance numérique définie pour les assertions.

---

## Epic PY-3 — Robustesse statistique & dominance

### Tickets

- **PY-3.1** Distribution de référence des DCA passifs (Monte Carlo + quantiles).
- **PY-3.2** Percentile de performance de la DCA Grid vs distribution passive.
- **PY-3.3** Dominance stochastique (perf + drawdown, ordre 1 simplifié).
- **PY-3.4** Comparaison au “best plausible passive” ex-ante.
- **PY-3.5** Stress tests de sensibilité des paramètres de grille.

### DoD

- Sortie statistique explicite (médiane, IQR, percentiles 5/25/50/75/95).
- Distinction claire entre tests ex-ante et analyses ex-post.
- Journalisation des seeds et des paramètres de simulation.

---

## Epic PY-4 — Analyse temporelle

### Tickets

- **PY-4.1** Rolling windows 3/5/10 ans.
- **PY-4.2** Détection de régimes de marché (règles simples, documentées).
- **PY-4.3** Stabilité de l’IRR en fenêtres glissantes.
- **PY-4.4** Détection des périodes de sous-performance structurelle.

### DoD

- Résultats indexés temporellement (par fenêtre, par régime).
- Méthode de segmentation des régimes documentée et figée.

---

## Epic PY-5 — Score synthétique

### Tickets

- **PY-5.1** Définir score composite (performance, IRR, drawdown, robustesse).
- **PY-5.2** Normaliser le score par rapport aux benchmarks passifs.
- **PY-5.3** Catégoriser l’edge (faible / moyen / fort).
- **PY-5.4** Tester sensibilité aux poids du score.

### DoD

- Poids et formules versionnés.
- Score interprétable (composants exposés, pas “black box”).

---

## Epic PY-6 — Reporting data contract (backend-ready pour Angular)

### Tickets

- **PY-6.1** Générer artefacts structurés (JSON/Parquet) : métriques, distributions, séries temporelles.
- **PY-6.2** Schéma contractuel versionné (`schema_version`).
- **PY-6.3** Exports synthèse par période testée.
- **PY-6.4** Pipeline reproductible (config + seed + hash dataset).

### DoD

- Contrat de données documenté et stable.
- Un run complet reproductible à l’identique.

---

## Epic PY-7 — Validation méthodologique

### Tickets

- **PY-7.1** Checklist anti data-snooping.
- **PY-7.2** Documentation exhaustive des hypothèses calendaires.
- **PY-7.3** Gel des règles benchmark ex-ante.
- **PY-7.4** Rapport de reproductibilité.

### Cadre d’audit vérifiable (à appliquer avant toute conclusion)

#### 1) Checklist anti-snooping (revue obligatoire)

- [ ] **Préréglage ex-ante**: paramètres de grille, dates, univers, métriques et seuils go/no-go gelés avant exécution.
- [ ] **Aucune optimisation post-hoc**: interdiction d’ajuster les paramètres sur la même période d’évaluation finale.
- [ ] **Split temporel respecté**: calibration/validation/test strictement ordonnés dans le temps (pas de mélange).
- [ ] **Zéro fuite de features**: tout indicateur utilise uniquement l’historique disponible à `t`.
- [ ] **Calendrier explicite**: jours non ouvrés, décalages d’exécution et fuseau horodatés et documentés.
- [ ] **Cashflows comparables**: budgets et règles d’investissement identiques entre stratégie DCA Grid et benchmarks.
- [ ] **Jeu de seeds traçable**: seeds Monte Carlo figées et archivées pour rerun bit-à-bit.
- [ ] **Versionning complet**: commit, spec, dataset hash, config runtime et artefacts signés dans le run manifest.
- [ ] **Comparaison canonique**: run de contrôle rejoué contre le baseline de référence et diff documenté.
- [ ] **Revue pairée**: validation indépendante signée avant communication d’un “edge”.

#### 2) Matrice des risques méthodologiques + mitigations

| Risque | Symptôme observable | Impact | Mitigation obligatoire | Evidence attendue |
|---|---|---|---|---|
| Data snooping | Perf excellente seulement après multiples tweaks | Surestimation de l’alpha | Gel ex-ante + journal des changements paramétriques | Changelog de specs + run manifest |
| Look-ahead bias | Signaux basés sur données futures | Résultats non réplicables en réel | Tests no-lookahead + audit des colonnes dérivées | Rapport tests + revue code |
| Survivorship bias | Univers actuel appliqué au passé | Biais positif structurel | Universe daté et versionné par période | Snapshot univers + dates effectives |
| Leakage calendrier | Exécutions sur dates impossibles | Biais opérationnel | Règles de trading calendar explicites | Trace d’exécution journalière |
| Overfitting de grille | Variance extrême hors échantillon | Robustesse faible | Validation OOS + stress tests paramètres | Table de sensibilité et quantiles |
| Non-reproductibilité | Résultats différents à spec identique | Audit impossible | Seeds, versions, hash datasets, artefacts figés | Manifest + checksum artefacts |

#### 3) Procédure de rerun reproductible

1. Sélectionner un identifiant de run canonique validé (baseline).  
2. Récupérer la **spec figée**, le **hash dataset**, le **commit git**, la version Python/poetry et les variables d’environnement requises.  
3. Exécuter le rerun avec la commande standardisée (`qe run-local`) sans modification locale de la spec.  
4. Générer les artefacts d’audit minimaux: `run_manifest.json`, `metrics.json`, `trades.parquet/csv`, `logs.txt`, `checksums.txt`.  
5. Comparer baseline vs rerun via diff déterministe: métriques principales, nombre de trades, distributions MC, checksums.  
6. Classer l’issue: **REPRODUCIBLE** (diffs dans tolérance), **DRIFT MINEUR**, ou **NON CONFORME** (investigation bloquante).  

#### 4) Critères go/no-go (formalisation)

- **GO** si toutes les conditions suivantes sont vraies:
  - Checklist anti-snooping: 100% validée.
  - Rerun canonique: statut `REPRODUCIBLE`.
  - Aucune fuite d’information détectée (tests + revue).
  - Performances DCA Grid supérieures au percentile ex-ante visé sur l’intervalle test.
  - Rapport méthodologique signé par 2 reviewers.
- **NO-GO** si au moins une condition suivante est observée:
  - Règles ex-ante modifiées après observation des résultats test.
  - Échec de reproductibilité non expliqué.
  - Divergence majeure entre benchmark calculé et baseline canonique.
  - Artefacts d’audit incomplets ou checksum manquant.
  - Fuite d’information non corrigée.

### DoD

- Audit trail complet (inputs, code version, paramètres, outputs).
- Revue pairée validant la non-fuite d’information.

---

## Annexe — Traçabilité exécutable des tickets EPIC-1..8

Cette annexe fixe la source de vérité pour l’audit « ticket -> tests -> commande ».  
Le détail ligne à ligne des sous-tickets PY-DCA est maintenu dans `tickets/active/PY-DCA-SUBTASKS-EXECUTION-PLAN.md` (Annexe A), avec commandes `pytest` prêtes CI.

### Commandes de validation canonique par EPIC

- **EPIC-1**: `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py tests/integration/test_dca_benchmark_specs.py`
- **EPIC-2**: `poetry run pytest -q tests/backtest/test_dca_metrics.py tests/api/test_runs_metrics_dca.py`
- **EPIC-3**: `poetry run pytest -q tests/stats/test_dca_robustness.py tests/integration/test_dca_robustness_pipeline.py`
- **EPIC-4**: `poetry run pytest -q tests/validate/test_dca_rolling_windows.py tests/integration/test_dca_rolling_regimes_pipeline.py`
- **EPIC-5**: `poetry run pytest -q tests/backtest/test_dca_score.py tests/integration/test_dca_score_sensitivity.py`
- **EPIC-6**: `poetry run pytest -q tests/io/test_dca_artifact_contract.py tests/api/test_dca_artifact_endpoints.py`
- **EPIC-7**: `poetry run pytest -q tests/test_backtest_engine_no_lookahead.py tests/integration/test_dca_robustness_pipeline.py`
- **EPIC-8**: `poetry run pytest -q tests/strategies/test_asset_universe_adapter.py tests/integration/test_dca_cross_universe_specs.py`

---

## 4) Roadmap d’implémentation côté Angular (priorité 2)

## Epic NG-1 — Ingestion & modèle de données UI

### Tickets

- **NG-1.1** Définir interfaces TypeScript alignées sur le contrat Python.
- **NG-1.2** Service d’accès aux artefacts/résultats (API ou fichiers).
- **NG-1.3** Gestion d’état (chargement, erreurs, version du run).

### DoD

- Affichage robuste en cas de données manquantes.
- Contrôle de compatibilité de version (`schema_version`).

---

## Epic NG-2 — Visualisations décisionnelles

### Tickets

- **NG-2.1** Boxplots perf vs DCA passifs.
- **NG-2.2** Position percentile de la grid dans la distribution.
- **NG-2.3** Courbes comparées de capital.
- **NG-2.4** Vues rolling windows (3/5/10 ans).
- **NG-2.5** Panneau drawdown + time to recovery.

### DoD

- Graphiques cohérents avec les sorties Python (tests snapshot visuels si possible).
- Lisibilité sur desktop + tablette.

---

## Epic NG-3 — Score & interprétation

### Tickets

- **NG-3.1** Carte score composite + détail des composantes.
- **NG-3.2** Badge edge (faible/moyen/fort).
- **NG-3.3** Explications méthodologiques intégrées (tooltips / help panel).

### DoD

- L’utilisateur comprend le “pourquoi” du score.
- Aucune métrique agrégée sans accès au détail source.

---

## Epic NG-4 — Reporting & export

### Tickets

- **NG-4.1** Export rapport synthèse (PDF/HTML/JSON).
- **NG-4.2** Export des graphiques et tableaux.
- **NG-4.3** Template de conclusion stratégique guidée.

### DoD

- Rapport exporté traçable (date, version, période, paramètres).

---

## 5) Ordonnancement recommandé (macro planning)

1. **Sprint 0 (Design/Spec)** : cadrage ex-ante + contrat de données + règles méthodo.
2. **Sprints 1–2 (Python Core)** : benchmarks + métriques + tests.
3. **Sprint 3 (Python Advanced)** : robustesse statistique + rolling windows + score.
4. **Sprint 4 (Python Data Contract)** : exports versionnés et reproductibilité.
5. **Sprints 5–6 (Angular)** : ingestion, dashboards, score, exports.
6. **Sprint 7 (Validation finale)** : audit méthodo + documentation + go/no-go.

---

## 6) Critères de décision produit (conclusion stratégique)

À la fin, la stratégie DCA Grid est qualifiée :

- **Alpha réel** si surperformance robuste, répétable, et conservée après pénalités de risque.
- **Confort psychologique** si amélioration surtout comportementale (drawdown perçu, volatilité vécue) sans edge statistique durable.

Conditions d’abandon recommandées :

- Sous-performance persistante sur rolling windows longues.
- Dominance non démontrée vs passifs ex-ante plausibles.
- Score composite dégradé sous un seuil minimal défini à l’avance.


---

## 7) Mutualisation multi-univers (ETF / Actions / Crypto)

Oui, l’approche doit être mutualisée : la logique DCA, les métriques et le protocole statistique sont largement transverses.  
Seules certaines briques doivent être spécialisées par classe d’actifs.

### 7.1 Ce qui est 100% mutualisable

- Moteur de cashflows DCA (mensuel/hebdo/randomisé/mid-month/turn-of-month).
- Moteur de métriques (perf normalisée, TWR, IRR/XIRR, max drawdown, time under water).
- Pipeline de robustesse (Monte Carlo calendrier, percentiles, dominance, stress tests).
- Data contract (`schema_version`, artefacts JSON/Parquet, reproductibilité).
- UX Angular de comparaison, score et reporting.

### 7.2 Ce qui doit être paramétré par univers

- Calendrier de marché (24/7 crypto vs jours ouvrés actions/ETF).
- Conventions d’exécution (lot minimal, fractionnable ou non, précision quantité/prix).
- Modèle de frais/slippage/financement (spot, broker equity, éventuels frais overnight).
- Qualité et fréquence des données (daily actions/ETF, intraday/24-7 crypto).
- Règles corporate actions (splits/dividendes pour actions/ETF).

### 7.3 Design recommandé (Python)

- Introduire un `AssetUniverseAdapter` avec implémentations : `etf`, `equity`, `crypto`.
- Le `StrategyRunner` consomme l’adapter (calendrier, execution rules, fees model, data policy).
- Les métriques restent universelles et ne dépendent pas directement de la classe d’actifs.
- Le reporting ajoute une dimension `universe` pour segmenter les résultats sans dupliquer le code.

### 7.4 Tickets additionnels (mutualisation)

- **PY-8.1** Créer l’interface `AssetUniverseAdapter` + 3 implémentations (ETF/Equity/Crypto).
- **PY-8.2** Ajouter tests de non-régression cross-universe (mêmes seeds, mêmes inputs, comparabilité).
- **PY-8.3** Versionner les règles univers (`universe_rules_version`) dans les artefacts.
- **NG-5.1** Ajouter un sélecteur d’univers dans l’UI et filtrage des vues.
- **NG-5.2** Ajouter une vue comparative inter-univers (score/risque/robustesse).

### 7.5 Règle méthodologique clé

- **Comparer une stratégie uniquement contre des benchmarks plausibles du même univers** (pas de conclusion “crypto > ETF” sans normalisation stricte du risque, des frais et du calendrier).

## 8) Mapping point d’entrée robustesse (clarification architecture)

Pour EPIC-3/P1, le **calcul canonique** des artefacts de robustesse DCA reste implémenté dans `quant_engine.stats.runner` via `compute_dca_robustness_artifacts`.

Afin d’éviter toute ambiguïté côté point d’entrée ticket `optimize/runner.py`, un pont explicite est exposé:

- `quant_engine.optimize.runner.compute_dca_robustness_from_stats(out_df, seed)`
- ce helper délègue directement à `quant_engine.stats.runner.compute_dca_robustness_artifacts(...)`

Conséquence: la logique métier de robustesse n’est pas dupliquée; `optimize` fournit uniquement un point d’accès aligné pour les workflows qui partent de l’optimisation.
