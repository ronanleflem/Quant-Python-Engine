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

### DoD

- Audit trail complet (inputs, code version, paramètres, outputs).
- Revue pairée validant la non-fuite d’information.

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
