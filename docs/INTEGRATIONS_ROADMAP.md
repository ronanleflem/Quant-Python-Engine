# Quant Engine – Integrations & Roadmap

## Intégrations actuelles
- **Lecture OHLCV (CSV/JSON)** : specs `ExperimentSpec` acceptent des chemins locaux pour charger les séries via le loader générique (`dataset_path`).【F:README.md†L7-L65】【F:docs/architecture_overview.md†L19-L36】
- **Lecture OHLCV (MySQL marketdata)** : datafeed `load_ohlcv_mysql` (SQLAlchemy) alimenté par `QE_MARKETDATA_MYSQL_URL`, options de lookup symbol/timeframe/chunking.【F:README.md†L74-L123】【F:src/quant_engine/datafeeds/mysql_feed.py†L45-L118】
- **Écriture des résultats** : persistance SQL (SQLite/MySQL) via `experiment_runs`, `market_stats`, `seasonality_*` + artefacts Parquet/JSON sur disque.【F:src/quant_engine/persistence/db.py†L48-L215】【F:src/quant_engine/optimize/runner.py†L78-L118】

## Interfaces front/back (Java/Angular)
- **Spécifications JSON** : contrats d’entrée pour API/CLI (`submit`, `stats.run`, `seasonality.run`), exemples fournis dans `specs/examples/*.json`.【F:src/quant_engine/api/app.py†L540-L698】【F:specs/examples/submit_mysql.json†L1-L49】
- **Endpoints de lecture** : `/runs`, `/runs/{id}`, `/runs/{id}/trials`, `/stats`, `/stats/summary`, `/stats/heatmap`, `/seasonality/profiles`, `/seasonality/runs`. Les réponses exposent JSON plat compatible clients Java/Angular.【F:src/quant_engine/api/app.py†L602-L812】
- **Conventions** : identifiants `run_id`, champs `spec_id/dataset_id`, schémas JSON normalisés (listes de colonnes + rows pour les stats, payloads `metrics` sérialisés).【F:src/quant_engine/api/app.py†L589-L720】【F:src/quant_engine/seasonality/runner.py†L160-L220】

## Roadmap proposée
- **Seasonality V2→V5** : élargir le set de dimensions/métriques (ex. edges spécifiques, corrélations multi-symboles) – *unknown (à définir)*.
- **Nouveaux datafeeds** : connecteur direct Binance (REST/WebSocket) pour bypass MySQL – *unknown (design à produire)*.
- **Parallélisation compute** : distribution des calculs seasonality/stats via multiprocessing ou Ray – *unknown (architecture à valider)*.
- **Dashboards Angular** : UI riche pour visualiser lifts, profils saisonniers, leaderboard d’optimisation – *unknown (design produit)*.
