# Getting started

Ce guide résume les premières étapes pour utiliser Quant Engine en local.

## Installation rapide

1. Installe les dépendances système (voir README).
2. Clone le dépôt et installe l'environnement Poetry :
   ```bash
   poetry install
   ```
3. Configure les variables d'environnement nécessaires (`QE_MARKETDATA_MYSQL_URL`, `DB_DSN`, etc.).

## Lancer l'API localement

```bash
poetry run uvicorn quant_engine.api.app:app --reload --app-dir src
```

## Lancer la CLI

Les commandes Typer sont exposées sous l'alias `qe` :

```bash
poetry run qe --help
```

## Essayer les filtres Volatilité/Tendance

Un exemple prêt à l'emploi est fourni dans `specs/filters_volatility_trend_example.json`.

```bash
poetry run qe stats run --spec specs/filters_volatility_trend_example.json
```

Ce scénario combine `adx`, `atr` et `ema_slope` pour illustrer les filtres de tendance et de volatilité.

## Essayer les filtres Volume/Profile

Les filtres pré-trade peuvent être évalués dans les workflows de statistiques. Un exemple complet est fourni dans `specs/filters_volume_profile_example.json`.

```bash
poetry run qe stats run --spec specs/filters_volume_profile_example.json
```

Le résultat présente les lifts calculés après application des filtres `volume_surge`, `vwap_side` et `poc_distance`. Lorsque la base `marketdata.levels` n'est pas accessible, les filtres reviennent automatiquement sur leurs fallbacks (ou renvoient `False`).

## Essayer les filtres Structure/ICT

Les filtres structurels et ICT disposent d'un exemple dédié dans `specs/filters_structure_ict_example.json`.

```bash
poetry run qe YOUR_COMMAND --spec specs/filters_structure_ict_example.json
```

Remplace `YOUR_COMMAND` par la commande de backtest/statistiques adaptée à ton workflow (ex. `stats run`).

## Essayer les filtres Seasonality/Time

Les filtres basés sur l'heure et la saisonnalité disposent d'un exemple dans `specs/filters_time_seasonality_example.json`.

```bash
poetry run qe YOUR_COMMAND --spec specs/filters_time_seasonality_example.json
```

Comme pour les autres exemples, remplace `YOUR_COMMAND` par la commande souhaitée (`stats run`, etc.).

## Essayer les filtres Statistiques/Probabilistes

Un exemple combinant les filtres de cette famille est fourni dans `specs/filters_stat_prob_example.json`.

```bash
poetry run qe YOUR_COMMAND --spec specs/filters_stat_prob_example.json
```

Adapte `YOUR_COMMAND` au workflow visé (par exemple `stats run`).

## Essayer les filtres Risk & Money Management

Les filtres de gestion du risque disposent d'un exemple complet dans `specs/filters_risk_mgmt_example.json`.

```bash
poetry run qe YOUR_COMMAND --spec specs/filters_risk_mgmt_example.json
```

Comme pour les autres scénarios, remplace `YOUR_COMMAND` par la commande correspondant à ton pipeline (`stats run`, backtest, etc.).


## Activer la feature Currency Strength (Option B)

Tu peux enrichir les runs avec `ccy_strength_base`, `ccy_strength_quote`, `ccy_strength_spread` via la section `features.currency_strength`.

Exemple :

```json
{
  "features": {
    "currency_strength": {
      "enabled": true,
      "lookback": 72,
      "majors": ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
    }
  }
}
```

Ce bloc fonctionne sur les workflows backtest/strategy runner et enrichit les données avant application des filtres.

## Utiliser la condition stats `currency_strength_regime` (Option C)

Dans une spec stats, tu peux classer le régime de force devise via :

```json
{
  "conditions": [
    {
      "name": "ccy_regime",
      "type": "currency_strength_regime",
      "params": {
        "spread_col": "ccy_strength_spread",
        "long_threshold": 0.2,
        "short_threshold": -0.2
      }
    }
  ]
}
```

Assure-toi que la colonne `ccy_strength_spread` est bien présente dans les données exploitées par le run stats.


## Lire les résultats Currency Strength (long vs short)

Interprétation des colonnes enrichies :

- `ccy_strength_base` : force relative de la devise de base.
- `ccy_strength_quote` : force relative de la devise cotée.
- `ccy_strength_spread = base - quote`.

Règle de lecture :
- `spread > 0` : contexte plutôt **long**.
- `spread < 0` : contexte plutôt **short**.
- `spread ~ 0` : contexte neutre, avantage faible.

Réaction opérationnelle suggérée :
- au-dessus d’un seuil `+T` (ex: `+0.2`) -> privilégier les longs,
- en dessous de `-T` (ex: `-0.2`) -> privilégier les shorts,
- entre `-T` et `+T` -> filtrer ou réduire l’exposition.

Cette logique correspond à la condition stats `currency_strength_regime` (`long` / `neutral` / `short`) pour vérifier quantitativement le comportement avant d’en faire une règle stricte en production.

