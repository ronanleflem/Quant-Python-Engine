# Workflow d'optimization

## Entree officielle d'optimisation (recommandee)

Il existe **deux** chemins d'optimisation dans le repo, avec des objectifs differents :

1) **`optimize.variants` (officiel/recommande)**  
   - Utilise par la CLI `qe strategy optimize` et `qe backtest optimize`.  
   - Supporte grid/random search, screening, promotion, deduplication, full-pass et artefacts
     detailles.  
   - Adapte aux specs backtest/strategy modernes (payloads riches, logs, sauvegardes).

2) **`optimize.runner` (runner simple/legacy)**  
   - Runner minimal base sur un espace de recherche fixe (EMA/ATR/R).  
   - Utilise comme **fallback** dans `qe run-local` quand le job manager n'est pas disponible.  
   - Ne couvre pas les fonctionnalites avancees (promotion, screening, dedup, etc.).

**Recommandation :** pour toute optimisation backtest/strategy, utilisez `optimize.variants`
via la CLI ou l'API. Le runner simple ne doit servir que pour des tests rapides ou le fallback local.
Ce document décrit le workflow d'optimization pour les specs de backtest et de strategy.

## Ce qui est stocké

Light data (toujours persistées par trial) :
- `trial_id`, `params`
- metrics agrégées (`sharpe`, `sortino`, `returnPct`, `maxDrawdownPct`, `winratePct`, `totalReturn`)
- valeur d'objective
- metadata de reproductibilite (`seed`, `dataset_id`, `timeframe`, `start`, `end`, `code_version`, `strategy_id`, `data_hash`, `config_hash`, `lib_versions`)

Heavy data (persistées uniquement pour les trials promus) :
- payload complet du trial (run + trades)

Les heavy payloads sont écrits dans `runs/optimize_backtest/promoted/` ou
`runs/optimize_strategy/promoted/` en `trial_{id}.json`.

### Artefacts compactes (light-heavy hybride)

Pour limiter le volume, tu peux activer un mode "compact" sur les payloads promus :

```json
{
  "optimization": {
    "artifacts": {
      "mode": "compact",
      "trade_sample_size": 50,
      "equity_max_points": 200
    }
  }
}
```

- `mode`: `full` (par d-faut), `compact`, ou `stats`
- `trade_sample_size`: nb de trades conserves en -chantillon
- `equity_max_points`: nb max de points d'equity conserves (downsample)
- `stats`: conserve uniquement `trades_stats` (pas de trades/equity bruts)

`trades_stats` inclut :
- quantiles (p10/p25/p50/p75/p90)
- wins / losses / winrate
- avg_win / avg_loss / rr_mean
- durees de trades (min/mean/p25/p50/p75/max en secondes)
- histogramme R (buckets)
- MAE/MFE si dispo dans les trades
- pnl par jour (agrégé par date d'exit)



### Promotion manager (niveaux d'artefacts)

Pour controler le volume disque, tu peux definir des niveaux d'artefacts par rang :

```json
{
  "optimization": {
    "promotion": {
      "top_k": 20,
      "levels": [
        { "max_rank": 3, "mode": "full" },
        { "max_rank": 10, "mode": "compact", "trade_sample_size": 50, "equity_max_points": 200 },
        { "max_rank": 20, "mode": "stats" }
      ]
    }
  }
}
```

- `levels` est applique sur la shortlist promue, ordonnee par objective.
- `max_rank` est inclusif (1..N).
- `mode` supporte `full`, `compact`, `stats`.
- Les overrides `trade_sample_size` / `equity_max_points` sont optionnels.

- Les niveaux sont utilises aussi pour le `full_pass` si tu n'as pas de overrides.
- Tu peux surcharger les niveaux du full_pass via `optimization.full_pass.artifacts.levels`.

## Promotion policy (top-K + constraints)

La promotion est contrôlée via `optimization.promotion` :

```json
{
  "optimization": {
    "objective": "sharpe",
    "promotion": {
      "top_k": 3,
      "hard_constraints": {
        "min_trades": 1,
        "max_drawdown_pct": 60,
        "min_winrate_pct": 20
      }
    }
  }
}
```

- `top_k` : nombre de meilleurs trials pour lesquels on garde les heavy payloads.
- `min_trades` : nombre minimum de trades (wins + losses).
- `max_drawdown_pct` : drawdown max autorisé (en %).
- `min_winrate_pct` : win rate minimum (en %).
- `min_return_pct` : return minimum (en %).
- `min_sharpe` : Sharpe minimum.
- `min_sortino` : Sortino minimum.
- `dedupe_distance` : filtre de diversité optionnel. Quand défini, un candidat
  est rejeté s'il est trop proche d'un trial déjà promu (distance calculée sur
  les paramètres normalisés via les bornes du search space).
- `behavior_distance` : filtre de diversité par comportement, basé sur des
  metrics de performance (voir ci-dessous).
Note: prefer `hard_constraints`/`soft_constraints` dans `promotion`; les cles flat restent acceptees pour compatibilite.



### Dedoublonnage deterministe (behavior)

Tu peux activer des logs expliquant pourquoi un trial est rejete (trop proche d'un autre) :

```json
{
  "optimization": {
    "promotion": {
      "dedupe_distance": 0.15,
      "behavior_distance": 0.2,
      "behavior_mode": "metrics",
      "behavior_metrics": ["returnPct", "maxDrawdownPct", "winratePct", "sharpe"],
      "behavior_bounds": {
        "returnPct": [-50, 200],
        "maxDrawdownPct": [0, 80],
        "winratePct": [0, 100],
        "sharpe": [-2, 5]
      },
      "log_dedupe": true
    }
  }
}
```

Logs :
- distance, trial remplace/rejete
- dimension dominante (param ou metric)

## Screening vs full run

Flow actuel :
1) Un passage sur tous les trials avec light storage (`trials.json`).
2) Promotion des top-K qui passent les constraints.
3) Persistance des heavy payloads uniquement pour les trials promus.

### Mode screening (raccourci)

Le screening permet d'accélérer l'optimization en utilisant un sous-ensemble
de données. Il est appliqué avant le calcul des filtres/signals.

```json
{
  "optimization": {
    "screening": {
      "enabled": true,
      "max_bars": 300,
      "max_trades": 25,
      "max_seconds": 2.0,
      "aggregate": "mean",
      "windows": [
        { "start": "2024-01-01", "end": "2024-06-30" },
        { "start": "2024-07-01", "end": "2024-12-31" }
      ]
    }
  }
}
```

Quand activé :
- seules les dernières `max_bars` sont utilisées pour le trial
- arrêt anticipé après `max_trades` trades complets (DCA/crypto grid)
- arrêt anticipé après `max_seconds` de temps de calcul
- `windows` (optionnel) exécute plusieurs sous-périodes et agrège l'objective
- `aggregate` contrôle l'agrégation (`mean`, `median`, `min`, `max`)
- le stockage reste light pour tous les trials


### Pruning "reel" (intra-window)

En plus du screening, tu peux couper un trial "mort" pendant l'execution :

```json
{
  "optimization": {
    "screening": {
      "enabled": true,
      "pruning": {
        "max_drawdown_pct": 25,
        "min_signals_after_bars": { "bars": 200, "min_signals": 1 }
      }
    }
  }
}
```

Regles appliquees pendant le backtest (intra-window) :
- `max_drawdown_pct` : stop immediat si le drawdown depasse la limite.
- `min_signals_after_bars` : stop si pas assez de signaux apres N barres.

Notes :
- actif uniquement si `screening.enabled = true`.
- applique aux backtests classiques et aux strategies DCA.



### Robustesse (pass 1)

Si tu utilises `windows`, tu peux forcer une robustesse minimale par fenetre :

```json
{
  "optimization": {
    "screening": {
      "windows": [
        { "start": "2025-01-01", "end": "2025-06-30" },
        { "start": "2025-07-01", "end": "2025-12-01" }
      ],
      "aggregate": "median",
      "min_window_objective": 0.1,
      "min_windows_passed": 2,
      "max_windows_failed": 0
    }
  }
}
```

- `min_window_objective`: seuil minimum par fenetre.
- `min_windows_passed`: nb minimum de fenetres au-dessus du seuil.
- `max_windows_failed`: nb max de fenetres en dessous du seuil.

## Mode full pass 2

Apres le screening + promotion, tu peux relancer uniquement la shortlist
sur l'historique complet :

```json
{
  "optimization": {
    "full_pass": {
      "enabled": true,
      "artifacts": {
        "mode": "full"
      }
    }
  }
}
```

- le full pass desactive le screening automatiquement
- les payloads sont -crits dans `runs/optimize_*/full_pass/`
- par d-faut, le full_pass reutilise `promotion.levels` pour les niveaux d'artefacts
- tu peux surcharger avec `optimization.full_pass.artifacts.levels`



### Full pass walk-forward (CV temporelle)

Tu peux transformer le full pass en walk-forward en fournissant des folds :

```json
{
  "optimization": {
    "full_pass": {
      "enabled": true,
      "aggregate": "median",
      "folds": [
        { "start": "2025-01-01", "end": "2025-06-30" },
        { "start": "2025-07-01", "end": "2025-12-01" }
      ],
      "artifacts": {
        "levels": [
          { "max_rank": 1, "mode": "full" },
          { "max_rank": 3, "mode": "compact", "trade_sample_size": 50, "equity_max_points": 200 }
        ]
      }
    }
  }
}
```

- `folds` : fenetres temporelles de validation.
- `aggregate` : aggregation robuste de l'objective par fold (`mean`, `median`, `min`, `max`).
- Chaque trial full_pass stocke les metrics par fold + l'aggregate.

## Freeze & refine (re-optimization ciblee)

Le mode refine relance une optimization sur un search space reduit, base sur
les meilleurs trials de la passe 1.

```json
{
  "optimization": {
    "refine": {
      "enabled": true,
      "top_k": 5,
      "shrink_pct": 0.5
    }
  }
}
```

- `top_k`: nb de meilleurs trials utilises pour estimer le centre
- `shrink_pct`: pourcentage de reduction de la plage initiale
- les ranges `min/max/step` sont reduites autour de la mediane
- les listes discretes sont reduites aux valeurs observees dans le top_k
- `freeze_keys`: liste de cles a ne pas reduire lors du refine
- `freeze_prefixes`: liste de prefixes a geler (ex: `strategy.params.grid`)



### Cache des calculs invariants (two-phase light)

Tu peux activer un cache des filtres/indicateurs pour eviter de recalculer
les memes series sur le meme dataset pendant l'optimization :

```json
{
  "optimization": {
    "cache_features": {
      "enabled": true,
      "max_items": 2048
    }
  }
}
```

- `max_items` limite la taille du cache en memoire.
- Le cache est scope par (symbol, dataset, screening) et par filtre+params.

## Log d'impact stockage

Le runner logue un resume du ratio heavy vs total trials :

```
Optimization storage impact: trials=1000 heavy_promoted=5 (0.5%) full_pass=5
```



### Debug on fail (payload compact)

Tu peux demander un dump compact quand un trial echoue ou retourne une objective non finie :

```json
{
  "optimization": {
    "debug_on_fail": {
      "enabled": true,
      "mode": "stats",
      "trade_sample_size": 50,
      "equity_max_points": 200
    }
  }
}
```

Les dumps sont ecrits dans `runs/optimize_*/debug_failures/`.

## Logs de configuration (traceability)

Au demarrage d'une optimization, le runner logue un resume complet du "cahier des charges" utilise :
- method, objective, nb de trials, params explores
- screening (windows, max_bars / max_trades / max_seconds)
- promotion (top_k + contraintes)
- behavior (mode, metrics, bins, clustering)
- artifacts (mode, samples) + full_pass
- refine (top_k, shrink_pct, freeze_keys/prefixes)

Fallbacks explicites logues :
- method inconnu -> grid
- behavior_mode inconnu -> metrics
- behavior_cluster.mode inconnu -> kmeans
- promotion.levels manquant -> fallback sur artifacts.mode

### Contraintes hard vs soft

Les contraintes hard (gates) eliminent un trial immediatement :

```json
{
  "optimization": {
    "promotion": {
      "hard_constraints": {
        "min_trades": 5,
        "min_winrate_pct": 15
      }
    }
  }
}
```

Les contraintes soft ajoutent une penalite dans l'objective :

```json
{
  "optimization": {
    "promotion": {
      "soft_constraints": {
        "max_drawdown_pct": {
          "threshold": 40,
          "direction": "above",
          "power": 2,
          "weight": -0.5
        }
      }
    }
  }
}
```

Notes :
- Les soft_constraints sont mergees avec `objective.penalties` (si presente).
- Les hard_constraints remplacent les anciennes cles flat `min_trades`, `max_drawdown_pct`, etc.



### Reproductibilite renforcee (hash + versions)

Les runs stockent maintenant :
- `config_hash` : hash de la spec complete.
- `data_hash` : hash du bloc `data`.
- `lib_versions` : versions des libs (pandas/numpy/requests/sqlalchemy/deltalake).

Cela permet de tracer exactement pourquoi un resultat change.

## Composite objective

Tu peux definir une objective composee avec des poids :

```json
{
  "optimization": {
    "objective": {
      "weights": {
        "sharpe": 1.0,
        "return_pct": 0.3,
        "max_drawdown_pct": -0.5
      }
    }
  }
}
```

Notes :
- les weights peuvent etre negatifes pour p-naliser un metric (ex: drawdown)
- si un metric est manquant, il est traite comme 0

### Penalites non lineaires (advanced)

Tu peux ajouter des penalites pour pousser l'optimizer a eviter certains regimes :

```json
{
  "optimization": {
    "objective": {
      "weights": {
        "sharpe": 1.0,
        "return_pct": 0.3
      },
      "penalties": {
        "max_drawdown_pct": {
          "threshold": 40,
          "direction": "above",
          "power": 2,
          "weight": -0.5
        }
      }
    }
  }
}
```

- `threshold`: point de depart de la penalite
- `direction`: `above` (par d-faut) ou `below`
- `power`: exponent pour renforcer la penalite
- `weight`: signe/poids applique a la penalite (negatif pour p-naliser)
## Diversite par comportement (promotion)

Pour eviter des variantes quasi identiques en performance, tu peux activer un
filtre de distance sur des metrics de comportement :

```json
{
  "optimization": {
    "promotion": {
      "behavior_distance": 0.2,
      "behavior_metrics": ["returnPct", "maxDrawdownPct", "winratePct", "sharpe"],
      "behavior_bounds": {
        "returnPct": [-50, 200],
        "maxDrawdownPct": [0, 80],
        "winratePct": [0, 100],
        "sharpe": [-2, 5]
      }
    }
  }
}
```

- `behavior_distance` : distance minimale (0-1) entre deux trials promus.
- `behavior_metrics` : liste de metrics utilises pour la distance.
- `behavior_bounds` : bornes [min,max] pour normaliser chaque metric.

Mode avance (histogramme de trades) :

```json
{
  "optimization": {
    "promotion": {
      "behavior_mode": "trades_hist",
      "behavior_distance": 0.2,
      "behavior_bins": [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    }
  }
}
```

- `behavior_mode`: `metrics` (par d-faut), `trades_hist` ou `equity_signature`
- `behavior_bins`: bornes des buckets pour l'histogramme (pnl_pct / r_multiple)

Mode avance (equity signature) :

```json
{
  "optimization": {
    "promotion": {
      "behavior_mode": "equity_signature",
      "behavior_distance": 0.2,
      "behavior_points": 20
    }
  }
}
```

- `behavior_mode`: `equity_signature` utilise un profil de performance cumul-
  (cumul- des pnl_pct) -chantillonne sur `behavior_points`

Tu peux basculer de l'un a l'autre en changeant simplement `behavior_mode`.

### Clustering par comportement

Apres la promotion, tu peux appliquer un clustering pour garder 1-2 candidats
par cluster (au lieu de garder des variantes tres proches) :

```json
{
  "optimization": {
    "promotion": {
      "behavior_cluster": {
        "enabled": true,
        "k": 3,
        "max_per_cluster": 2
      }
    }
  }
}
```

- `k`: nombre de clusters
- `max_per_cluster`: nb max de candidats retenus par cluster

Option : `k` peut etre `auto` (sqrt du nombre de candidats promus, min 2).

Clustering robuste (auto):
- `auto_mode`: `silhouette` (d-faut) ou `inertia`
- `fallback_mode`: mode de secours si auto echoue
- `min_k` / `max_k`: bornes pour la recherche auto
- `iterations`: iterations du k-means

Mode avance (DBSCAN-like):
- `mode`: `dbscan`
- `eps`: rayon de voisinage (distance)
- `min_samples`: nombre minimum de voisins pour former un cluster

## Guide rapide (checklist)

1) Definir `search_space` (discret ou min/max/step).
2) Choisir `objective` (simple ou composite).
3) Activer `screening` si besoin (max_bars / max_trades / max_seconds).
4) Configurer `promotion` (top_k + contraintes).
5) Lancer la commande d'optimization.

## Exemple complet (strategy DCA)

```json
{
  "optimization": {
    "method": "grid",
    "objective": {
      "weights": {
        "sharpe": 1.0,
        "return_pct": 0.3,
        "max_drawdown_pct": -0.5
      }
    },
    "search_space": {
      "strategy.filters[0].params.window": { "min": 20, "max": 100, "step": 10 },
      "strategy.params.grid[0].dd": [-15.0, -20.0, -25.0],
      "strategy.params.tp_sl.rules[0].tp_pct": { "min": 10.0, "max": 30.0, "step": 5.0 }
    },
    "screening": {
      "enabled": true,
      "max_bars": 300,
      "max_trades": 25,
      "max_seconds": 2.0,
      "aggregate": "mean",
      "windows": [
        { "start": "2024-01-01", "end": "2024-06-30" },
        { "start": "2024-07-01", "end": "2024-12-31" }
      ]
    },
    "promotion": {
      "top_k": 3,
      "min_trades": 1,
      "max_drawdown_pct": 80,
      "min_winrate_pct": 10,
      "dedupe_distance": 0.15
    }
  }
}
```







### Analyse de sensibilite (post-optimization)

Tu peux activer une analyse simple sur les meilleurs trials :

```json
{
  "optimization": {
    "sensitivity": {
      "enabled": true,
      "top_k": 20
    }
  }
}
```

Sortie :
- `correlations`: corr?lation param?tre ? objective (top_k).
- `stable_params`: param?tres peu sensibles (valeurs quasi constantes).

## Exemples JSON (optimization)

- `specs/examples/optimization/backtest_eurusd_m1_optimize_pruning_levels.json`
- `specs/examples/optimization/backtest_eurusd_m1_optimize_cache_features.json`
- `specs/examples/optimization/backtest_eurusd_m1_optimize_debug_on_fail.json`
- `specs/examples/optimization/backtest_eurusd_m1_optimize_fullpass_levels.json`
- `specs/examples/optimization/backtest_eurusd_m1_optimize_fullpass_folds.json`
- `specs/examples/optimization/strategy_dca_equity_optimize_levels.json`
- `specs/examples/optimization/strategy_dca_equity_optimize_levels_hard_soft.json`
- `specs/examples/optimization/strategy_dca_equity_optimize_dedupe_logs.json`
- `specs/examples/optimization/strategy_dca_equity_optimize_retention.json`
- `specs/examples/optimization/strategy_dca_equity_optimize_repro_hash.json`
- `specs/examples/optimization/strategy_dca_equity_optimize_sensitivity.json`
- `specs/examples/optimization/strategy_dca_equity_optimize_windows_median.json`



### Retention (budget stockage)

Tu peux limiter le nombre de runs conserves et purger les artefacts lourds :

```json
{
  "optimization": {
    "storage": {
      "retention": {
        "enabled": true,
        "keep_last_runs": 5,
        "keep_best_runs": 3,
        "mode": "heavy_only",
        "dry_run": false
      }
    }
  }
}
```

- `keep_last_runs` : conserve les N runs les plus recents.
- `keep_best_runs` : conserve les N meilleurs runs par (strategy_id, dataset_id).
- `mode`: `heavy_only` (purge `promoted/` + `full_pass/`) ou `full` (supprime le run complet).
- `dry_run`: log sans suppression.

## Roadmap "niveau pro" (priorites)

P0 (ROI fort, faible complexite) :
- Pruning reel intra-window (drawdown / manque de signaux).
- Promotion Manager avec niveaux d'artefacts (NONE/STATS/EQUITY/TRADES_SAMPLE/FULL).
- Robustesse en pass 1 (aggregate median/min + must-pass par window).

P1 (qualite de selection) :
- Walk-forward/CV temporelle en pass 2 + metrics par fold.
- Constraints hard vs soft (gates + penalites dans l'objective).
- Dedoublonnage comportemental deterministe + logs de rejet explicites.

P2 (scalabilite 50k+ trials) :
- Budget stockage + retention automatique par run.
- Cache des calculs invariants (features) + two-phase compute.
- Reproductibilite beton (config_hash + data_hash + versions libs).

P3 (nice to have) :
- Debug_on_fail (dump compact en cas d'exception/NaN).
- Analyse de sensibilite post-optimization (importance params, freeze auto).



## Recap (cahier des charges)

P0
- Pruning reel intra-window (drawdown / manque de signaux).
- Promotion Manager avec niveaux d'artefacts (NONE/STATS/EQUITY/TRADES_SAMPLE/FULL).
- Robustesse pass 1 (aggregate median/min + must-pass par window).

P1
- Walk-forward / CV temporelle en pass 2 + metrics par fold.
- Contraintes hard vs soft (gates + penalites dans l'objective).
- Dedoublonnage comportemental deterministe + logs de rejet explicites.

P2
- Budget stockage + retention automatique par run.
- Cache des calculs invariants (two-phase light).
- Reproductibilite renforcee (config_hash + data_hash + versions libs).

P3
- Debug_on_fail (dump compact si exception/NaN).
- Analyse de sensibilite post-optimization (correlations + params stables).

## Resultats

- `runs/optimize_* /trials.json` : tous les trials (light data)
- `runs/optimize_* /summary.json` : meilleur trial + metadata + promoted
- `runs/optimize_* /promoted/trial_{id}.json` : payload complet des promus
