# Workflow d'optimization

Ce document décrit le workflow d'optimization pour les specs de backtest et de strategy.

## Ce qui est stocké

Light data (toujours persistées par trial) :
- `trial_id`, `params`
- metrics agrégées (`sharpe`, `sortino`, `returnPct`, `maxDrawdownPct`, `winratePct`, `totalReturn`)
- valeur d'objective
- metadata de reproductibilite (`seed`, `dataset_id`, `timeframe`, `start`, `end`, `code_version`)

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

- `mode`: `full` (par defaut), `compact`, ou `stats`
- `trade_sample_size`: nb de trades conserves en echantillon
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

## Promotion policy (top-K + constraints)

La promotion est contrôlée via `optimization.promotion` :

```json
{
  "optimization": {
    "objective": "sharpe",
    "promotion": {
      "top_k": 3,
      "min_trades": 1,
      "max_drawdown_pct": 60,
      "min_winrate_pct": 20,
      "min_return_pct": 0,
      "min_sharpe": 0.2,
      "min_sortino": 0.2,
      "dedupe_distance": 0.15
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
- les payloads sont ecrits dans `runs/optimize_*/full_pass/`

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

## Log d'impact stockage

Le runner logue un resume du ratio heavy vs total trials :

```
Optimization storage impact: trials=1000 heavy_promoted=5 (0.5%) full_pass=5
```

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

## Remaining work

- Ajouter un early-stop (max trades, max time) et du sub-window sampling.
- Ajouter des métadonnées explicites (dataset_id / code_version) dans les trials.
- Ajouter des promotion policies alternatives (objective composite, clustering par comportement).
- Ajouter des artefacts compressés optionnels (equity curve summary, trade stats only).

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
- les weights peuvent etre negatifes pour penaliser un metric (ex: drawdown)
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
- `direction`: `above` (par defaut) ou `below`
- `power`: exponent pour renforcer la penalite
- `weight`: signe/poids applique a la penalite (negatif pour penaliser)
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

- `behavior_mode`: `metrics` (par defaut), `trades_hist` ou `equity_signature`
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

- `behavior_mode`: `equity_signature` utilise un profil de performance cumule
  (cumule des pnl_pct) echantillonne sur `behavior_points`

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
- `auto_mode`: `silhouette` (defaut) ou `inertia`
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

## Resultats

- `runs/optimize_* /trials.json` : tous les trials (light data)
- `runs/optimize_* /summary.json` : meilleur trial + metadata + promoted
- `runs/optimize_* /promoted/trial_{id}.json` : payload complet des promus
