# Cache OHLC et filtres (TTL + métriques)

## Vue d'ensemble

Le moteur utilise deux caches mémoire pour limiter les recalculs :

1. **Cache OHLC** dans `strategies.runner` (chargement des données).
2. **Cache des filtres** dans `filters.utils` (résultats des masques).

Chaque cache est borné (LRU) et dispose d'un TTL afin d'éviter toute croissance
illimitée. Les entrées expirées sont purgées lors des accès et des écritures.
Des métriques simples (hits/misses/expirations/evictions, taille, TTL, max)
sont loguées à chaque exécution de backtest ou application de filtres pour
faciliter le suivi.

## Cache OHLC

Configuration côté `data` (ou directement par instrument) :

```json
{
  "data": {
    "ohlc_cache": {
      "enabled": true,
      "ttl_seconds": 900,
      "max_items": 256
    }
  }
}
```

Alias acceptés : `cache_ohlc` ou `cache`.

## Cache des filtres

Configuration côté optimisation (déjà utilisée pour activer le cache) :

```json
{
  "optimization": {
    "cache_features": {
      "enabled": true,
      "ttl_seconds": 900,
      "max_items": 2048
    }
  }
}
```

Le TTL et la taille maximale peuvent être ajustés par stratégie ; sinon les
valeurs par défaut s'appliquent.
