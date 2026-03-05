# Runbook opératoire — Market Intelligence (recompute/inspect + cache)

## 1) Objectif

Ce runbook décrit l'exploitation quotidienne des features Market Intelligence :

- recalcul ciblé (`features recompute`),
- inspection rapide (`features inspect`),
- maintenance et invalidation du cache,
- diagnostic incident (checklist prête à l'emploi).

Ce document complète l'architecture cible décrite dans `docs/architecture_market_intelligence_refactor.md`.

---

## 2) Pré-requis

- CLI installée : `qe`.
- Environnement Python du projet actif.
- Données OHLCV disponibles pour le symbole/timeframe visé.
- Si usage API distant : variable `QE_API_BASE_URL` correctement positionnée.

Checks rapides :

```bash
qe --help
python -m quant_engine.cli.main --help
```

---

## 3) Commandes opératoires features

> Référence d'interface cible (MI refactor) :
>
> - `qe features recompute --feature-set mi_v1 --symbol BTCUSDT --tf 1h --from ... --to ...`
> - `qe features inspect --feature-set mi_v1 --symbol BTCUSDT --tf 1h`

### 3.1 Recompute (recalcul)

Usage standard :

```bash
qe features recompute \
  --feature-set mi_v1 \
  --symbol BTCUSDT \
  --tf 1h \
  --from 2025-01-01T00:00:00Z \
  --to   2025-01-31T23:59:59Z
```

But opératoire :

- forcer un recalcul après évolution pipeline/version,
- recalculer une fenêtre temporelle précise,
- réaligner la donnée après correction source.

Bonnes pratiques :

- lancer d'abord sur une fenêtre courte avant un backfill large,
- vérifier explicitement `feature_set`, `tf` et bornes temporelles,
- conserver dans le ticket incident la fenêtre recalculée.

### 3.2 Inspect (inspection)

Usage standard :

```bash
qe features inspect \
  --feature-set mi_v1 \
  --symbol BTCUSDT \
  --tf 1h
```

But opératoire :

- valider la présence du snapshot attendu,
- contrôler rapidement `feature_version` et colonnes de features,
- comparer un symbole sain vs symbole en anomalie.

Points à vérifier lors de l'inspection :

- cohérence `symbol` / `timeframe`,
- version de features (`feature_version`) alignée avec le déploiement,
- présence des blocs `features`, `regimes`, `liquidity`.

---

## 4) Maintenance cache et invalidation

## 4.1 Ce qui est caché

Le service MI lit/écrit un snapshot via une clé logique :

`(feature_set, symbol, timeframe, version)`

Cette clé est celle du `InMemoryFeatureStore`.

## 4.2 Quand invalider

Invalider (ou forcer recompute) dans ces cas :

1. bump de version pipeline/features,
2. correction des données OHLCV source,
3. divergence constatée entre `inspect` et résultat attendu,
4. corruption suspectée d'entrée cache (valeurs incohérentes, trous).

## 4.3 Méthodes d'invalidation

### A. Invalidation ciblée (recommandée)

Supprimer uniquement la clé concernée, puis relancer `recompute` sur la fenêtre utile.

Principe :

- éviter un flush global coûteux,
- limiter l'impact sur les autres symboles/timeframes.

### B. Flush global (dernier recours)

À utiliser seulement si :

- bug transversal de versionnement,
- pollution cache généralisée.

Toujours documenter : cause, périmètre, heure, opérateur.

## 4.4 Rappels cache connexes (hors MI direct)

Le projet maintient aussi :

- un cache OHLC TTL/LRU,
- un cache filtres TTL/LRU (`optimization.cache_features`).

En cas de doute sur des résultats incohérents, vérifier aussi ces deux caches applicatifs.

---

## 5) Troubleshooting

## 5.1 Symptôme: `inspect` vide / clé absente

Causes probables :

- `feature_set`/`tf` incorrect,
- version différente de celle attendue,
- entrée évincée ou jamais calculée.

Actions :

1. relancer `recompute` sur petite fenêtre,
2. revalider la combinaison `(feature_set, symbol, tf, version)`,
3. élargir progressivement la fenêtre.

## 5.2 Symptôme: valeurs de features anormales

Causes probables :

- données OHLCV en entrée dégradées,
- recalcul partiel sur fenêtre insuffisante,
- mélange de versions en cache.

Actions :

1. comparer avec un symbole témoin,
2. invalider la clé ciblée,
3. relancer `recompute` avec bornes explicites.

## 5.3 Symptôme: latence élevée au recalcul

Causes probables :

- plage temporelle trop large en une passe,
- contention/charge infra,
- cascade de miss cache.

Actions :

1. découper en batches temporels,
2. surveiller la saturation worker/API,
3. exécuter hors pics de charge.

---

## 6) Checklist incident (copier/coller ticket)

- [ ] Incident qualifié (symptôme + périmètre symbole/timeframe).
- [ ] Commande `inspect` exécutée et sortie archivée.
- [ ] Version feature (`feature_version`) vérifiée.
- [ ] Hypothèse cache validée/infirmée.
- [ ] Invalidation ciblée effectuée si nécessaire.
- [ ] `recompute` exécuté sur fenêtre courte puis complète.
- [ ] Contrôle post-fix réalisé (inspect + comparaison témoin).
- [ ] Ticket enrichi (horodatage, commandes, opérateur, résultat).

---

## 7) Validation manuelle (cohérence doc/commandes)

Procédure minimale de validation doc :

1. vérifier la présence des commandes de référence dans la doc d'architecture,
2. vérifier la CLI disponible localement via `qe --help`,
3. consigner les écarts éventuels (ex. sous-commande non exposée dans ce build) avant exécution en production.


## 8) Commandes de non-régression et preuves

Exécuter les checks suivants avant validation d'un changement MI/legacy:

```bash
poetry run pytest -q tests/api
poetry run pytest -q tests/architecture/test_import_rules.py
poetry run pytest -q tests/integration/test_kpi_non_regression_mi.py
poetry run pytest -q
```

À archiver dans la PR/ticket:

- commande exécutée,
- statut (pass/fail),
- extrait de sortie (ou lien artifact CI),
- horodatage et commit SHA.


## 9) Release gate hardening (PY-MI-5.11)

### 9.1 Commande canonique de reproduction locale

```bash
poetry run pytest -q
```

Cette commande est la référence pour reproduire la gate CI de non-régression avant release.

### 9.2 Checklist “go prod” (bloquante)

- [ ] `poetry run pytest -q` vert (suite complète).
- [ ] `poetry run pytest -q tests/architecture/test_import_rules.py` vert.
- [ ] `poetry run pytest -q tests/integration/test_kpi_non_regression_mi.py` vert.
- [ ] Aucun test flaky observé sur les runs de validation release.
- [ ] Warnings restants triés et explicitement acceptés dans le ticket release.
- [ ] Gate CI confirmée bloquante sur échec (`.github/workflows/ci.yml`, step `Run full regression gate (pytest)`).
- [ ] Preuves archivées: commandes, statuts, durée, commit SHA, lien artifact CI.

### 9.3 Rollback plan (instabilité)

En cas d'instabilité de dernière minute:

1. isoler le(s) test(s) non déterministe(s) avec un marquage temporaire explicite,
2. ouvrir un ticket technique dédié (cause probable, impact, propriétaire, ETA),
3. documenter la justification de l'isolation dans la release,
4. rétablir la gate complète dès correction.
