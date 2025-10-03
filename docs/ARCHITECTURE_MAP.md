# Quant Engine – Architecture Map

## API FastAPI
| Endpoint | Méthode | Description | Entrée/Sortie | Référence |
|----------|---------|-------------|---------------|-----------|
| `/submit` | POST | Lance une optimisation synchronisée et enregistre le run en mémoire. | Body = spec JSON validée via Pydantic → `{ "id": run_id }`. | 【F:src/quant_engine/api/app.py†L540-L555】 |
| `/status/{job_id}` | GET | Vérifie l’état (`completed`, `unknown`, …) d’un job soumis. | Path param `job_id` → `{ "status": ... }`. | 【F:src/quant_engine/api/app.py†L557-L566】 |
| `/result/{job_id}` | GET | Récupère le résultat complet d’un job (paramètres, métriques, artefacts). | Path param `job_id` → `{ "result": {...} }`. | 【F:src/quant_engine/api/app.py†L568-L576】 |
| `/stats/run` | POST | Calcule un run Market Stats synchrone. | Body = `StatsSpec` → `{ "status": "completed" }`. | 【F:src/quant_engine/api/app.py†L578-L587】 |
| `/stats/result` | GET | Renvoie la dernière table de stats calculée. | `{ "result": {columns, rows} }`. | 【F:src/quant_engine/api/app.py†L589-L600】 |
| `/stats` | GET | Liste les stats persistées filtrables (symbol, event, target, split, significance). | Query params → liste de lignes enrichies (p_hat, lifts, FDR). | 【F:src/quant_engine/api/app.py†L602-L622】 |
| `/stats/summary` | GET | Agrège les stats par condition/target et calcule Wilson CI. | Query params (symbol/timeframe/event) → tableau agrégé. | 【F:src/quant_engine/api/app.py†L624-L648】 |
| `/stats/heatmap` | GET | Retourne les bins ordonnés (train/test) pour une condition donnée. | Query params `symbol,timeframe,event,target,condition_name`. | 【F:src/quant_engine/api/app.py†L650-L662】 |
| `/stats/top` | GET | Top-k patterns classés par lift fréquentiste ou bayésien. | Query `k`, `method`, `significant_only`. | 【F:src/quant_engine/api/app.py†L664-L678】 |
| `/seasonality/run` | POST | Exécute un profil saisonnalité complet (compute + persistance). | Body = `SeasonalitySpec` → résumé JSON. | 【F:src/quant_engine/api/app.py†L680-L688】 |
| `/seasonality/optimize` | POST | Lance l’optimisation Optuna dédiée saisonnalité. | Body = `SeasonalitySpec` → `{best_value, best_params, ...}`. | 【F:src/quant_engine/api/app.py†L690-L698】 |
| `/seasonality/profiles` | GET | Paginer les profils saisonniers stockés (filtrés par symbol/dim/metrics). | Query (metrics optionnelles) → liste de profils + metrics JSON. | 【F:src/quant_engine/api/app.py†L700-L720】 |
| `/seasonality/runs` | GET | Historique des runs saisonnalité (status, best_summary). | Query (status/spec_id/dataset_id). | 【F:src/quant_engine/api/app.py†L722-L740】 |
| `/seasonality/runs/{run_id}` | GET | Détail d’un run saisonnalité (404 si absent). | Path `run_id`. | 【F:src/quant_engine/api/app.py†L742-L753】 |
| `/runs` | GET | Liste des runs d’optimisation persistés (status, objective, dates). | Query `status/date_from/date_to`. | 【F:src/quant_engine/api/app.py†L755-L772】 |
| `/runs/{run_id}` | GET | Détails d’un run (metrics agrégés, folds, params). | Path `run_id`. | 【F:src/quant_engine/api/app.py†L774-L786】 |
| `/runs/{run_id}/trials` | GET | Leaderboard des essais Optuna. | Query `order_by`, pagination. | 【F:src/quant_engine/api/app.py†L788-L801】 |
| `/runs/{run_id}/metrics` | GET | Vue agrégée et par fold des métriques. | Path `run_id`. | 【F:src/quant_engine/api/app.py†L803-L812】 |

## CLI Typer
| Commande | Rôle | Paramètres clés | Référence |
|----------|------|-----------------|-----------|
| `qe run-local --spec` | Exécuter une spec en local (JobManager ou runner direct). | `--spec path/to/spec.json`. | 【F:src/quant_engine/cli/main.py†L23-L59】 |
| `qe submit --spec` | Soumettre la spec à l’API HTTP. | URL API fixe `http://127.0.0.1:8000`. | 【F:src/quant_engine/cli/main.py†L61-L105】 |
| `qe stats run --spec` | Calculer des Market Stats localement et afficher le top. | Option `--spec`. | 【F:src/quant_engine/cli/main.py†L107-L147】 |
| `qe stats show --symbol --event --target` | Lire les stats persistées depuis l’API. | `--method {freq|bayes}`, `--limit`, `--significant-only`. | 【F:src/quant_engine/cli/main.py†L149-L199】 |
| `qe seasonality run --spec` | Exécuter un profil saisonnalité local. | `--spec`. | 【F:src/quant_engine/cli/main.py†L201-L226】 |
| `qe seasonality optimize --spec` | Boucle Optuna saisonnalité. | `--spec`. | 【F:src/quant_engine/cli/main.py†L228-L252】 |
| `qe seasonality profiles` | Consommer l’endpoint `/seasonality/profiles`. | `--metrics`, `--limit`. | 【F:src/quant_engine/cli/main.py†L254-L304】 |
| `qe seasonality compare --symbols A --symbols B --dim` | Comparer les lifts saisonniers de deux symboles. | `--timeframe`, `--measure`. | 【F:src/quant_engine/cli/main.py†L306-L368】 |
| `qe runs list --status` | Liste les runs d’optimisation. | `--limit`. | 【F:src/quant_engine/cli/main.py†L370-L402】 |
| `qe runs show RUN_ID` | Affiche les détails (metrics, best params) d’un run. | `run_id`. | 【F:src/quant_engine/cli/main.py†L404-L437】 |

## Modules principaux
- `stats/`
  - `events.py` : événements (k consecutives, choc ATR, breakouts).【F:src/quant_engine/stats/events.py†L7-L39】
  - `conditions.py` : contextes HTF trend, tertiles de volatilité, session.【F:src/quant_engine/stats/conditions.py†L8-L44】
  - `targets.py` : outcomes `up_next_bar`, `continuation_n`, `time_to_reversal`.【F:src/quant_engine/stats/targets.py†L10-L33】
  - `estimators.py` : Wilson CI, Beta-Binomial, Benjamini–Hochberg, agrégations.【F:src/quant_engine/stats/estimators.py†L19-L180】
  - `runner.py` : orchestration du pipeline stats (chargement data, apply events/conditions/targets, persistance).【F:src/quant_engine/stats/runner.py†L1-L220】
- `seasonality/`
  - `compute.py` : construction des profils (dimensions hour/dow/etc., agrégats direction/retour).【F:docs/seasonality_reference.md†L9-L83】【F:src/quant_engine/seasonality/compute.py†L1-L80】
  - `profiles.py` : formatage, comparaison, corrélation des lifts.【F:src/quant_engine/seasonality/profiles.py†L1-L112】
  - `runner.py` : orchestration (chargement dataset, calcul, export Parquet/DB).【F:src/quant_engine/seasonality/runner.py†L1-L220】
  - `optimize.py` : optimisation Optuna (recherche des meilleures combinaisons saisonnières).【F:src/quant_engine/seasonality/optimize.py†L1-L90】
- `backtest/` : moteur bar-based, calculs metrics (Sharpe, Sortino, maxDD).【F:docs/architecture_overview.md†L19-L43】【F:src/quant_engine/backtest/metrics.py†L1-L56】
- `tpsl/` : règles de stops/targets (ATR multiples, take-profit en R).【F:docs/architecture_overview.md†L25-L33】【F:src/quant_engine/tpsl/rules.py†L1-L32】
- `optimize/` : runner Optuna, job manager, persistence des trials.【F:docs/architecture_overview.md†L19-L36】【F:src/quant_engine/optimize/runner.py†L1-L120】
- `validate/` : splitters Walk-Forward, embargo.【F:docs/architecture_overview.md†L23-L31】【F:src/quant_engine/validate/splitter.py†L1-L38】
- `persistence/` : accès DB (SQLite/MySQL via SQLAlchemy), modèles dataclass, dépôts.【F:src/quant_engine/persistence/db.py†L8-L215】【F:src/quant_engine/persistence/models.py†L1-L58】
- `datafeeds/` : chargement OHLCV via CSV/Parquet/MySQL (`load_ohlcv_mysql`).【F:src/quant_engine/datafeeds/mysql_feed.py†L1-L118】

## Base de données (schéma quant)
- `experiment_runs` : statut du run, spec/dataset, objectif, timestamps.【F:src/quant_engine/persistence/db.py†L48-L69】
- `run_metrics` : métriques agrégées et par fold (Unique par run/fold/metric).【F:src/quant_engine/persistence/db.py†L70-L91】
- `trials` : historique Optuna (params JSON, objectif, métriques auxiliaires).【F:src/quant_engine/persistence/db.py†L92-L119】
- `market_stats` : résultats stats conditionnelles (p_hat, lifts, FDR, bornes temporelles).【F:src/quant_engine/persistence/db.py†L120-L167】
- `seasonality_profiles` : profils saisonniers, lifts et métriques sérialisées.【F:src/quant_engine/persistence/db.py†L168-L206】
- `seasonality_runs` : suivi des runs saisonnalité (status, best_summary).【F:src/quant_engine/persistence/db.py†L207-L215】
- Migrations : gérées via Alembic (déclaré côté dépendances, migrations custom dans `persistence/`).【F:pyproject.toml†L9-L20】【F:docs/architecture_overview.md†L11-L36】
