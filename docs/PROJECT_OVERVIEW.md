# Quant Engine – Project Overview

## Rôle
- **Moteur d'optimisation** : orchestration FastAPI/CLI pour lancer des backtests, optimiser une stratégie Optuna et historiser les runs.【F:README.md†L7-L65】【F:src/quant_engine/api/app.py†L14-L112】
- **Analyse statistique** : calcul de statistiques de marché (probabilités, intervalles de confiance, filtrage FDR) et profils de saisonnalité persistés en base.【F:README.md†L102-L171】【F:src/quant_engine/stats/estimators.py†L19-L180】

## Stack technique
- **Langage** : Python 3.11 (gestion via Poetry).【F:pyproject.toml†L1-L27】
- **API** : FastAPI + Pydantic pour l’exposition REST et la validation des specs.【F:pyproject.toml†L9-L20】【F:src/quant_engine/api/app.py†L536-L754】
- **CLI** : Typer pour `quant-engine` et sous-commandes runs/stats/seasonality.【F:pyproject.toml†L9-L20】【F:src/quant_engine/cli/main.py†L12-L212】
- **Optimisation** : Optuna pour la recherche d’hyperparamètres.【F:pyproject.toml†L9-L20】【F:README.md†L7-L21】
- **Persistance** : SQLAlchemy + Alembic, artefacts Parquet/JSON, MLflow optionnel.【F:pyproject.toml†L9-L20】【F:docs/architecture_overview.md†L11-L36】
- **Colonnes & calculs** : Polars/NumPy/Pandas pour transformations vectorisées.【F:pyproject.toml†L9-L20】【F:docs/seasonality_reference.md†L1-L74】

## Installation & exécution
1. **Installer les dépendances** : `poetry install` après avoir configuré Python 3.11 via pyenv/Poetry.【F:README.md†L12-L47】
2. **Variables d’environnement** :
   - `QE_DB_URL` : *unknown* (la base actuelle s’appuie sur `DB_DSN`).【F:src/quant_engine/config.py†L15-L38】
   - `QE_MARKETDATA_MYSQL_URL` : DSN MySQL requis pour lire les OHLCV via datafeed dédié.【F:README.md†L74-L123】【F:src/quant_engine/datafeeds/mysql_feed.py†L45-L118】
3. **Lancer l’API** : `poetry run uvicorn quant_engine.api.app:app --reload --app-dir src`.【F:README.md†L49-L54】
4. **Utiliser la CLI** :
   - `poetry run quant-engine run-local --spec path/to/spec.json`
   - `poetry run quant-engine submit --spec path/to/spec.json`
   - `poetry run quant-engine runs list --status running`
   - `poetry run quant-engine runs show RUN_ID`
   - `poetry run quant-engine stats run --spec specs/stats_run.json`
   - `poetry run quant-engine stats show --symbol EURUSD --event k_consecutive --target up_next_bar`
   - `poetry run quant-engine seasonality run --spec specs/seasonality.json`
   - `poetry run quant-engine seasonality optimize --spec specs/seasonality.json`【F:README.md†L56-L101】【F:src/quant_engine/cli/main.py†L23-L211】

## Arborescence condensée
```
src/quant_engine/
  api/            # FastAPI (routes, schémas)
  cli/            # Typer CLI
  backtest/       # Moteur bar-based + métriques
  optimize/       # Optuna runner & job manager
  stats/          # Events, conditions, targets, estimators
  seasonality/    # Compute, profiles, optimisation
  tpsl/           # Règles take-profit/stop-loss
  validate/       # Walk-forward & validation
  persistence/    # SQLAlchemy/Alembic + repositories
  datafeeds/      # Connecteurs CSV/MySQL
  io/             # Gestion d’artefacts et IDs
```
【F:src/quant_engine/__init__.py†L1-L1】【F:docs/architecture_overview.md†L11-L23】【F:src/quant_engine/persistence/db.py†L8-L115】
