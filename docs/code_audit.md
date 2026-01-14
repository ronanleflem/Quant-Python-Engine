# Audit fonctionnel approfondi (21 points) + priorités

> Objectif : approfondir les 21 axes d’audit identifiés et proposer un plan de tâches priorisées.
> Ce document ne modifie pas le code ; il synthétise les observations et recommandations.

## Top 3 tâches immédiates (priorité + faible chevauchement)

1. **Unifier le parsing des specs** (créer un point d’entrée unique et réutilisé par backtest/strategy/live).  
   - _Pourquoi en premier ?_ Réduit les divergences fonctionnelles entre modules et limite les bugs de config.  
   - _Faible conflit_ : touche principalement `core.spec` + points d’appel.  
   - _Réf_ : duplication actuelle dans le backtest runner.【F:src/quant_engine/core/spec.py†L110-L188】【F:src/quant_engine/backtest/runner.py†L42-L83】

2. **Normaliser parsing timestamp/session + nettoyage du dataset loader** (supprimer duplications et rendre la logique homogène).  
   - _Pourquoi ?_ Garantit une base de données propre et stable, impact direct sur backtests/stats/seasonality.  
   - _Faible conflit_ : centré sur `core.dataset` et fonctions utilitaires.  
   - _Réf_ : duplication et logique dispersée dans `core.dataset`.【F:src/quant_engine/core/dataset.py†L24-L218】

3. **Clarifier et documenter le point d’entrée officiel d’optimisation** (variants vs runner simple).  
   - _Pourquoi ?_ Évite l’usage du runner minimal pour des runs de prod ; réduit la confusion d’architecture.  
   - _Faible conflit_ : changement documentaire principalement, éventuellement refactor léger de CLI.  
   - _Réf_ : coexistence `optimize.runner` et `optimize.variants`.【F:src/quant_engine/optimize/runner.py†L1-L94】【F:src/quant_engine/optimize/variants.py†L1-L120】

---

## 1) Configuration & settings

**Constat**
- Le module `config` fournit un loader léger basé sur les variables d’environnement avec cache LRU, mais la validation des types est minimale (conversion booléenne simple, aucune validation forte de format).【F:src/quant_engine/config.py†L17-L44】
- Plusieurs modules consomment des variables d’environnement non centralisées (ex. `QE_JAVA_BASE_URL`, `QE_MARKETDATA_MYSQL_URL`, `QE_JAVA_LIVE_URL`), sans contrat unique de configuration, ce qui fragilise la cohérence entre CLI/API/live.【F:src/quant_engine/integrations/java_client.py†L10-L140】【F:src/quant_engine/live/runner.py†L43-L116】

**Risques/impact**
- Des erreurs de configuration silencieuses peuvent entraîner des comportements inattendus (ex. mauvais DSN, env absent, booléen mal interprété).

**Recommandations**
- Ajouter une couche de validation (schéma typed) et centraliser les variables d’environnement critiques.

---

## 2) Spécifications (spec) & parsing

**Constat**
- Le parsing des specs est partiellement dupliqué (ex. `DataSpec` côté `core.spec` et re-parsing côté `backtest.runner`).【F:src/quant_engine/core/spec.py†L110-L188】【F:src/quant_engine/backtest/runner.py†L42-L83】
- Validation assez légère : peu de constraints sur types, valeurs ou champs requis en profondeur.

**Risques/impact**
- Incohérences entre modules (backtest/strategy/live) quand des champs évoluent.

**Recommandations**
- Unifier le parsing (un seul point d’entrée) et ajouter des validations explicites pour les champs majeurs.

---

## 3) Chargement dataset & formats

**Constat**
- `load_dataset` gère CSV/JSON/MySQL mais le parsing des timestamps et l’injection de `session` sont dispersés, avec un `raise` dupliqué en fin de fichier (code mort).【F:src/quant_engine/core/dataset.py†L24-L218】
- Les sessions sont déterminées par tranches horaires fixes, sans tenir compte des calendriers de marché/fuseaux exacts.【F:src/quant_engine/core/dataset.py†L38-L48】

**Risques/impact**
- Incohérences de normalisation (sessions/timestamps) et faiblesse sur marchés avec horaires non standards.

**Recommandations**
- Harmoniser parsing & session logic, supprimer les duplications et rendre le mapping de sessions configurable.

---

## 4) Datafeeds MySQL

**Constat**
- Chargement MySQL correct mais sans gestion de retry/backoff et sans gestion d’indexing explicite. Chunking basé sur minutes peut être lourd sur gros volumes.【F:src/quant_engine/datafeeds/mysql_feed.py†L60-L152】

**Risques/impact**
- Risque de timeouts sur de grandes périodes, et latence élevée sans optimisations DB.

**Recommandations**
- Documenter les index requis + ajouter un retry/backoff réseau + pagination batchée côté DB.

---

## 5) API (FastAPI)

**Constat**
- Jobs et stats stockés en mémoire (_jobs, _last_stats), perte au restart. Endpoints synchrones pour tâches lourdes (stats/levels/seasonality).【F:src/quant_engine/api/app.py†L33-L99】

**Risques/impact**
- Non-résilience et blocage possible pour des workloads importants.

**Recommandations**
- Externaliser le job storage (DB/Redis) et implémenter un mode async pour les runs lourds.

---

## 6) CLI

**Constat**
- URL API hardcodée (127.0.0.1:8000) et utilisation de `urllib` sans retry/backoff. Pas de client HTTP partagé centralisé.【F:src/quant_engine/cli/main.py†L64-L176】

**Risques/impact**
- Faible robustesse en environnement distribué (latence/timeouts/erreurs transitoires).

**Recommandations**
- Paramétrer l’URL API + implémenter un client partagé (requests.Session) avec retry/backoff.

---

## 7) Backtest (engine + runner)

**Constat**
- Le moteur supporte un cas long-only simple avec stop/TP basés sur ATR; pas de short, pas d’exécution intrabar réaliste, et modèle de coût très simplifié.【F:src/quant_engine/backtest/engine.py†L12-L165】
- Le runner ne supporte qu’un signal de type `ema_cross`. Les filtres sont un mécanisme séparé, appliqué après coup via `apply_filter_stack`.【F:src/quant_engine/backtest/runner.py†L161-L270】

**Risques/impact**
- Limitation de l’expressivité des stratégies, divergence possible avec le live.

**Recommandations**
- Étendre le moteur (short, slippage avancé, gestion multi-orders) et généraliser les signaux via registry.

---

## 8) Performance & payloads (backtest + DCA)

**Constat**
- Le calcul des métriques backtest repose sur returns % simples, sans ajustement risk-free, sans annualisation ou normalisation timeframe.【F:src/quant_engine/performance/backtest_builder.py†L30-L151】
- Le builder DCA ignore les signaux sans `cycle_id` (ils sont dropped), ce qui peut supprimer des signaux réels avant agrégation en trades complets.【F:src/quant_engine/performance/dca_builder.py†L107-L137】

**Risques/impact**
- Perte d’information et métriques incomplètes; possible biais dans le résumé des trades.

**Recommandations**
- Ajouter une gestion explicite des signaux sans `cycle_id` (log/flag/flux séparé) + normalisation des métriques.

---

## 9) Strategies runner

**Constat**
- Cache OHLC global sans TTL ni limites explicites; pas de stratégie de refresh configurée.【F:src/quant_engine/strategies/runner.py†L33-L244】
- Fallback de sources (Delta/MySQL/Java) très pratique, mais pas de contrôle sur la fraîcheur des données.

**Risques/impact**
- Données potentiellement obsolètes et consommation mémoire non maîtrisée.

**Recommandations**
- Ajouter TTL/éviction + options de refresh par stratégie ou par run.

---

## 10) Filters

**Constat**
- Catalogue de filtres riche avec validation et cache LRU mais sans observabilité (hit/miss, taille) ni configuration globale claire.【F:src/quant_engine/filters/utils.py†L71-L98】【F:src/quant_engine/filters/utils.py†L125-L189】

**Risques/impact**
- Difficulté à diagnostiquer les coûts et les ralentissements.

**Recommandations**
- Instrumenter le cache (metrics) et centraliser la configuration de limites/taille.

---

## 11) Levels

**Constat**
- Le fill parcourt désormais les niveaux actifs par pages (keyset) et applique les updates par lot, ce qui réduit la pression mémoire sur des univers étendus.【F:src/quant_engine/levels/runner.py†L32-L95】

**Risques/impact**
- La volumétrie extrême reste sensible au coût DB (index et batch_size à ajuster).

**Recommandations**
- Ajuster `batch_size` côté repo si besoin et monitorer la latence DB.

---

## 12) Stats

**Constat**
- Construction long-form via `iterrows`, potentiellement très lente sur gros volumes. Seuils globaux fixes (`N_MIN=300`).【F:src/quant_engine/stats/runner.py†L37-L217】

**Risques/impact**
- Performances dégradées et manque de flexibilité pour différents datasets.

**Recommandations**
- Vectoriser (ou passer à Polars) et rendre les seuils paramétrables via spec.

---

## 13) Seasonality

**Constat**
- Dépendance stricte à `polars`, pas de fallback léger. Persistance DB mais absence de mécanisme de retry/reprise pour run interrompu.【F:src/quant_engine/seasonality/runner.py†L10-L236】

**Risques/impact**
- Robustesse réduite si `polars` absent ou si le run échoue en cours.

**Recommandations**
- Fallback pandas minimal ou erreur plus actionnable + mécanisme de reprise/monitoring du run.

---

## 14) Optimisation

**Constat**
- Il existe un module d’optimisation avancé pour backtest/strategy via `optimize.variants` (grid/random/refine, dédup, promotion, screening).【F:src/quant_engine/optimize/variants.py†L1-L120】【F:src/quant_engine/optimize/variants.py†L1805-L2391】
- Le runner simple (`optimize.runner`) est encore très basique, mais il n’est pas le seul mécanisme disponible.【F:src/quant_engine/optimize/runner.py†L1-L94】

**Risques/impact**
- Possible confusion entre le “runner simple” et la logique avancée. Documentation ou conventions d’usage insuffisantes.

**Recommandations**
- Clarifier le point d’entrée “officiel” d’optimisation (backtest/strategy) et documenter les workflows attendus.

---

## 15) Validation (walk-forward)

**Constat**
- `_add_months` simplifie le jour à 28 pour éviter les issues de calendrier, ce qui peut fausser les splits sur données réelles.【F:src/quant_engine/validate/splitter.py†L8-L43】

**Risques/impact**
- Splits peu fidèles aux cycles calendaires.

**Recommandations**
- Utiliser un calcul de calendrier plus précis (pandas offsets).

---

## 16) Live runner

**Constat**
- Pipeline live basé sur polling MySQL, état en mémoire, pas de métriques latence/backlog, pas de stratégie de reprise robuste.【F:src/quant_engine/live/runner.py†L40-L212】

**Risques/impact**
- Risque d’accumulation de retard et de dérive entre bar traité et bar réel.

**Recommandations**
- Ajouter observabilité (bar-age, backlog), backoff et snapshot d’état.

---

## 17) Intégration Java

**Constat**
- Appels HTTP simples via `requests` sans session partagée, pas de retry/backoff global.【F:src/quant_engine/integrations/java_client.py†L48-L130】

**Risques/impact**
- Fragilité sur erreurs transitoires réseau.

**Recommandations**
- Centraliser un `Session` avec retry/backoff et timeouts standardisés.

---

## 18) Persistence (SQLite)

**Constat**
- Schéma créé en code sans migration/versionning. Approche légère adaptée aux tests mais fragile pour évolutions DB réelles.【F:src/quant_engine/persistence/db.py†L1-L236】

**Risques/impact**
- Migration manuelle difficile en production.

**Recommandations**
- Ajouter un mécanisme de migration (ex. Alembic) ou au moins un versionning simple.

---

## 19) Execution / Broker

**Constat**
- Broker minimal (single position, no sizing, no fees). Utile pour tests mais trop simple pour simuler des stratégies réelles.【F:src/quant_engine/execution/broker.py†L1-L35】

**Risques/impact**
- Backtests irréalistes, écart avec exécution live.

**Recommandations**
- Ajouter sizing, frais, slippage, et multi-positions.

---

## 20) Signals & TP/SL

**Constat**
- `EmaCross` est simple, sans gestion explicite de warmup (valeurs instables pour début de série).【F:src/quant_engine/signals/ema_cross.py†L10-L18】
- TP/SL ATR ne gère pas explicitement les cas ATR manquants / index hors borne.【F:src/quant_engine/tpsl/rules.py†L7-L35】

**Risques/impact**
- Signaux et stops potentiellement erronés au début des séries.

**Recommandations**
- Ajouter une logique de warmup et des garde-fous pour ATR manquants.

---

## 21) IO / Artifacts

**Constat**
- Artifacts écrits en JSON/Parquet sans compression; pas d’option pour formats volumineux ou streaming.【F:src/quant_engine/io/artifacts.py†L23-L61】

**Risques/impact**
- Overhead I/O et stockage volumineux sur gros runs.

**Recommandations**
- Ajouter un mode compression et config d’output format.

---

# Tâches les plus prioritaires (proposées)

## Priorité 0 (stabilité & cohérence)
1. **Unifier le parsing des specs** (éviter les duplications backtest/core).【F:src/quant_engine/core/spec.py†L110-L188】【F:src/quant_engine/backtest/runner.py†L42-L83】
2. **Normaliser parsing timestamp/session + nettoyage du `core.dataset`** (supprimer duplications et standardiser les formats).【F:src/quant_engine/core/dataset.py†L24-L218】
3. **Clarifier le point d’entrée officiel d’optimisation** (documenter `optimize.variants` vs `optimize.runner`).【F:src/quant_engine/optimize/variants.py†L1-L120】【F:src/quant_engine/optimize/runner.py†L1-L94】

## Priorité 1 (résilience & performance)
4. **Persist job/status API + async jobs** pour stats/levels/optimization.【F:src/quant_engine/api/app.py†L33-L99】
5. **Retry/backoff pour intégrations HTTP & CLI** (session partagée, timeouts).【F:src/quant_engine/integrations/java_client.py†L48-L130】【F:src/quant_engine/cli/main.py†L64-L176】
6. **Cache OHLC/filters avec TTL + métriques** (limites explicites).【F:src/quant_engine/strategies/runner.py†L33-L244】【F:src/quant_engine/filters/utils.py†L71-L98】

## Priorité 2 (fonctionnel métier)
7. **Backtest engine plus réaliste** (short, slippage avancé, intrabar).【F:src/quant_engine/backtest/engine.py†L12-L165】
8. **Gestion explicite des signaux sans `cycle_id`** (flag + flux séparé).【F:src/quant_engine/performance/dca_builder.py†L107-L137】
9. **Vectorisation Stats** (perf sur gros datasets).【F:src/quant_engine/stats/runner.py†L37-L217】

## Priorité 3 (améliorations avancées)
10. **Validation calendrier plus fidèle** (splits mensuels).【F:src/quant_engine/validate/splitter.py†L8-L43】
11. **Live runner observability** (latence, backlog, state snapshot).【F:src/quant_engine/live/runner.py†L40-L212】
12. **Compression artifacts** (gros outputs).【F:src/quant_engine/io/artifacts.py†L23-L61】
