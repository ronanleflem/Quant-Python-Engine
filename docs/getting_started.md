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
