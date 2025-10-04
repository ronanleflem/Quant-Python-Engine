![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)

# Quant Engine

## Présentation
Moteur d’optimisation et de backtest basé sur une spécification JSON, prenant en charge EMA/VWAP, TP/SL, Walk Forward Analysis, Optuna, MySQL et MLflow.

Affichage Markdown :  Ctrl + Shift + V
## Installation
```bash 
# Installer les dépendances nécessaires
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git

# Installer pyenv (via curl)
curl https://pyenv.run | bash

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

source ~/.bashrc
# ou
source ~/.zshrc

pyenv install 3.11.9
pyenv global 3.11.9

python --version
# Doit afficher Python 3.11.9

pyenv local 3.11.9
poetry env use python3.11
poetry install

export $(grep -v '^#' .env | xargs)
```

## 📖 Documentation
- [Getting started](docs/getting_started.md)
- [Filtres pré-trade](docs/filters.md)
- [Seasonality – Dimensions & Métriques](docs/seasonality_reference.md)

## Lancer l'API
```bash
poetry run uvicorn quant_engine.api.app:app --reload --app-dir src
```

## CLI
- **run-local**
  ```bash
  poetry run quant-engine run-local --spec path/to/spec.json
  ```
- **submit**
  ```bash
  poetry run quant-engine submit --spec path/to/spec.json
  ```
- **runs list**
  ```bash
  poetry run quant-engine runs list --status running
  ```
- **runs show**
  ```bash
  poetry run quant-engine runs show RUN_ID
  ```

## Configuration `.env`
Copier `.env.example` vers `.env` et ajuster :
```env
DB_DSN=sqlite:///.db/quant.db
DB_ECHO=false
MLFLOW_TRACKING_URI=http://localhost:5000
```
Pour les tests rapides, la persistance locale utilise SQLite (`.db/quant.db`). Configure les accès MySQL via `QE_MARKETDATA_MYSQL_URL` lorsque tu veux lire les OHLCV depuis ton instance Spring.


## Docker Compose
```bash
docker compose up -d
poetry run alembic upgrade head
```

## Qualité
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Endpoints clés
- `POST /submit`
- `GET /status/{id}`
- `GET /result/{id}`
- `GET /runs`
- `GET /runs/{id}`
- `GET /runs/{id}/trials`
- `GET /runs/{id}/metrics`

### Tests rapides de l'API

Prerequis:
- API lancée localement (`poetry run uvicorn quant_engine.api.app:app --reload --app-dir src`).
- Exemples de spécifications accessibles dans `specs/examples`.
- Sous PowerShell, conserver les commandes `curl` sur une seule ligne ou utiliser l'accent grave `` ` `` pour un retour chariot. Sous Bash/Zsh, les continuations `\` fonctionnent comme d'habitude.

#### Optimisation (EMA cross demo)
```bash
curl.exe -X POST http://127.0.0.1:8000/submit -H "Content-Type: application/json" --data-binary @specs/examples/submit_local.json
# -> {"id":"RUN_ID"}
```

```bash
curl.exe http://127.0.0.1:8000/status/RUN_ID
curl.exe http://127.0.0.1:8000/result/RUN_ID | python -m json.tool
```

```bash
python -m json.tool summary.json

> Exemple MySQL : `specs/examples/submit_mysql.json` (requiert `QE_MARKETDATA_MYSQL_URL`).
```

#### Statistiques (in-memory + SQLite)
```bash
curl.exe -X POST http://127.0.0.1:8000/stats/run -H "Content-Type: application/json" --data-binary @specs/examples/stats_run.json
```

```bash
curl.exe http://127.0.0.1:8000/stats/result | python -m json.tool
```

```bash
curl.exe "http://127.0.0.1:8000/stats?symbol=EURUSD&timeframe=M1&target=up_next_bar&page_size=20" | python -m json.tool
```

```bash
curl.exe "http://127.0.0.1:8000/stats/top?symbol=EURUSD&timeframe=M1&k=5" | python -m json.tool
```

```bash
curl.exe "http://127.0.0.1:8000/stats/summary?symbol=EURUSD&timeframe=M1" | python -m json.tool
```

```bash
curl.exe "http://127.0.0.1:8000/stats/heatmap?symbol=EURUSD&timeframe=M1&event=k_consecutive&target=up_next_bar&condition_name=session" | python -m json.tool
```

#### Saisonalité (profil + optimisation Optuna)
```bash
curl.exe -X POST http://127.0.0.1:8000/seasonality/run -H "Content-Type: application/json" --data-binary @specs/examples/seasonality_run.json | python -m json.tool
```

```bash
curl.exe -X POST http://127.0.0.1:8000/seasonality/optimize -H "Content-Type: application/json" --data-binary @specs/examples/seasonality_optimize.json | python -m json.tool
```

```bash
curl.exe "http://127.0.0.1:8000/seasonality/runs?spec_id=demo_seasonality_run&page_size=5" | python -m json.tool
```

```bash
curl.exe "http://127.0.0.1:8000/seasonality/profiles?symbol=EURUSD&spec_id=demo_seasonality_run&page_size=5" | python -m json.tool
```

```bash
python -m json.tool runs/seasonality_demo/fold_0/summary.json
python -m json.tool runs/seasonality_opt_demo/fold_0/summary.json
```



### Recap par module

| Module | Commandes curl (README) | Variables a exporter | Notes |
| --- | --- | --- | --- |
| Optimisation (`/submit`) | bloc Optimisation (`curl -X POST .../submit`, `status`, `result`) | `DB_DSN=sqlite:///.db/quant.db` (persistance) ; `QE_MARKETDATA_MYSQL_URL` si lecture MySQL | Prend en charge `dataset_path` (JSON/CSV) ou `data.mysql`. Persistance locale = fichiers `summary.json`/`trials.parquet`. |
| Statistiques (`/stats/*`) | bloc Statistiques (`/stats/run`, `result`, `stats`, `stats/top`, `stats/summary`, `stats/heatmap`) | `DB_DSN=sqlite:///.db/quant.db` (obligatoire) ; `QE_MARKETDATA_MYSQL_URL` si lecture MySQL | Résultats écrits dans `.db/quant.db` (`market_stats`). Prévoir `session_id` si condition `session`. |
| Saisonalite (`/seasonality/*`) | bloc Saisonalite (`/seasonality/run`, `/seasonality/optimize`, `/seasonality/runs`, `/seasonality/profiles`) | `DB_DSN=sqlite:///.db/quant.db` ; `QE_MARKETDATA_MYSQL_URL` si lecture MySQL ; `polars` installé | Persistance dans `.db/quant.db` (`seasonality_*`) + artefacts `runs/`. `seasonality_optimize` s'appuie sur Optuna. |

Pense a exporter `DB_DSN=sqlite:///.db/quant.db` avant de lancer l'API, puis `QE_MARKETDATA_MYSQL_URL` vers `mysql+pymysql://restadmin:ronanronan77@127.0.0.1:3306/restdb?charset=utf8mb4` si tu veux lire tes OHLCV MySQL.

#### Nettoyage des artefacts
- Les API statistiques et saisonnalité écrivent dans `.db/quant.db`. Supprimer ce fichier pour repartir de zéro (`rm .db/quant.db` ou `Remove-Item .db/quant.db`).
- Les artefacts locaux sont générés dans `runs/` et les fichiers `summary.json` / `trials.parquet` à la racine. Supprimer ces éléments si nécessaire.
- `seasonality_run` et `seasonality_optimize` requièrent `polars` (installé via Poetry) et utilisent les scénarios réduits fournis.

Note: `specs/examples/submit_local.json` référence le mini jeu de données `specs/examples/data/eurusd_m1_sample.json`. Les paramètres de validation (`train_months=0`, `test_months=1`, `folds=1`) sont volontairement minimalistes pour produire un fold sur ce jeu réduit. Remplacez-les (et les données) pour vos tests avancés et veillez à enregistrer vos JSON en UTF-8 sans BOM (PowerShell: `-Encoding utf8NoBOM`).




## Market Stats

### Exemple de spécification

```json
{
  "data": {
    "dataset_path": "data/eurusd_m1_2025H1.csv",
    "symbols": ["EURUSD"],
    "timeframe": "M1",
    "start": "2025-01-01",
    "end": "2025-06-30"
  },
  "events": [
    { "name": "k_consecutive", "params": { "k": 2, "direction": "up" } }
  ],
  "conditions": [
    { "name": "htf_trend", "params": { "tf_multiplier": 60, "ema_period": 50 } },
    { "name": "vol_tertile", "params": { "window": 14 } },
    { "name": "session", "params": { "col": "session_id" } }
  ],
  "targets": [
    { "name": "up_next_bar", "params": {} },
    { "name": "continuation_n", "params": { "n": 3, "direction": "up" } }
  ],
  "validation": { "scheme": "walk_forward", "train_months": 2, "test_months": 1, "folds": 3, "embargo_days": 2 },
  "artifacts": { "out_dir": "runs/stats_eurusd_m1_2025H1", "save_equity": false, "save_trades": false },
  "persistence": { "store_trades_in_db": false, "store_equity_in_db": false }
}
```

### Charger OHLCV depuis MySQL (schéma marketdata)

```json
{
  "data": {
    "mysql": {
      "env_var": "QE_MARKETDATA_MYSQL_URL",
      "schema": "restdb",
      "table": "candle",
      "symbol_col": "symbol_id",
      "ts_col": "date",
      "open_col": "open",
      "high_col": "high",
      "low_col": "low",
      "close_col": "close",
      "volume_col": "volume",
      "timeframe_col": "timeframe",
      "extra_where": null,
      "chunk_minutes": 0,
      "symbol_lookup_table": "symbol",
      "symbol_lookup_symbol_col": "symbol",
      "symbol_lookup_id_col": "id"
    },
    "symbols": ["EURUSD"],
    "timeframe": "M1",
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  }
}
```

> Exemple complet à adapter : `specs/examples/stats_run_mysql.json` (lecture `restdb` via `symbol_lookup_table`).


Variables d’environnement à définir :

```bash
export DB_DSN='sqlite:///.db/quant.db'                                  # Persistance locale
export QE_MARKETDATA_MYSQL_URL='mysql+pymysql://restadmin:ronanronan77@127.0.0.1:3306/restdb?charset=utf8mb4'  # Lecture OHLCV

# sous PowerShell
$env:DB_DSN='sqlite:///.db/quant.db'
$env:QE_MARKETDATA_MYSQL_URL='mysql+pymysql://restadmin:ronanronan77@127.0.0.1:3306/restdb?charset=utf8mb4'
poetry run uvicorn quant_engine.api.app:app --reload --app-dir src
```

Index recommandés dans `marketdata` :

- `(symbol, ts)` lorsque la table est partitionnée par timeframe (ex. `ohlcv_m1`).
- `(symbol, timeframe, ts)` lorsqu’une table unique regroupe plusieurs timeframes.

Les timestamps (`ts`) doivent être en UTC.

### CLI

```bash
poetry run quant-engine stats run --spec path/to/stats_spec.json
poetry run quant-engine stats show --symbol EURUSD --event k_consecutive --target up_next_bar --timeframe M1 --limit 20
```

### API

- `POST /stats/run`
- `GET /stats/result`
- `GET /stats`

Tests (curl):
```bash
# 1. Lancer un run statistiques
curl.exe -X POST http://127.0.0.1:8000/stats/run -H "Content-Type: application/json" --data-binary @specs/examples/stats_run.json
```

```bash
# 2. Recuperer le dernier resultat en memoire
curl http://127.0.0.1:8000/stats/result
```

```bash
# 3. Interroger les stats persistees
curl "http://127.0.0.1:8000/stats?symbol=EURUSD&event=k_consecutive&target=up_next_bar&limit=5"
```

Note: `/stats/result` renvoie le resultat de la derniere execution locale, tandis que `/stats` lit les donnees persistees en base. La spec d'exemple (`specs/examples/stats_run.json`) supprime la validation multi-fen0tre et n'utilise pas de conditions pour rester compatible avec le mini dataset; adaptez validation/conditions 0 votre cas r0el.

## Seasonality Backtest

➡️ Voir [Seasonality – Dimensions & Métriques](docs/seasonality_reference.md) pour la liste complète des dimensions et métriques disponibles.

### Exemple de spécification

```json
{
  "data": {
    "dataset_path": "data/eurusd_m1_2025H1.csv",
    "symbols": ["EURUSD"],
    "timeframe": "M1",
    "start": "2025-01-01",
    "end": "2025-06-30"
  },
  "profile": {
    "by_hour": true,
    "by_dow": true,
    "by_month": false,
    "measure": "direction",
    "ret_horizon": 1,
    "min_samples_bin": 300
  },
  "signal": {
    "method": "threshold",
    "threshold": 0.54,
    "dims": ["hour", "dow"],
    "combine": "and"
  },
  "execution": { "slippage_bps": 0.5, "fee_bps": 0.2 },
  "risk": { "position_sizing": "fixed_fraction", "risk_per_trade": 0.005 },
  "tp_sl": {
    "stop": { "type": "fixed_atr", "atr_mult": 1.5 },
    "take_profit": { "type": "r_multiple", "r_values": [1, 2, 3, 4, 5] }
  },
  "validation": { "scheme": "walk_forward", "train_months": 2, "test_months": 1, "folds": 3, "embargo_days": 2 },
  "artifacts": { "out_dir": "runs/seasonality_eurusd_m1_2025H1", "save_equity": true, "save_trades": true },
  "persistence": { "store_trades_in_db": false, "store_equity_in_db": false }
}
```

### CLI

```bash
poetry run qe seasonality run --spec specs/eurusd_m1_seasonality.json  # spec complète
```

> Exemple MySQL prêt à l'emploi (à adapter) : `specs/examples/seasonality_run_mysql.json`.


```bash
# Lister les profils saisonnalité persistés en filtrant sur des métriques conditionnelles
poetry run qe seasonality profiles --symbol EURUSD --metrics run_len_up_mean,p_breakout_up
```

```bash
# Comparer deux symboles sur une dimension et afficher la corrélation des lifts
poetry run qe seasonality compare --symbols EURUSD DXY --dim hour --timeframe M1
```

### API

- `POST /seasonality/run`
- `POST /seasonality/optimize`
- `GET /seasonality/profiles`
- `GET /seasonality/runs`
- `GET /seasonality/runs/{run_id}`

Tests (curl):
```bash
# 1. Lancer un backtest saisonnalite
curl -X POST http://127.0.0.1:8000/seasonality/run -H "Content-Type: application/json" --data-binary @specs/examples/seasonality_run.json
```

```bash
# 2. Consulter les profils stockes
curl "http://127.0.0.1:8000/seasonality/profiles?symbol=EURUSD&page_size=5"
```

Note: l'exemple rapide utilise `specs/examples/seasonality_run.json`, calibré pour le mini dataset JSON (et nécessite `polars`). Les routes `/seasonality/profiles` et `/seasonality/runs*` supposent une base renseignee via Alembic.

### Dimensions & signal

- `by_hour`, `by_dow`, `by_month` activent les agrégations par heure, jour de semaine ou mois pour les profils.
- `measure` choisit la métrique : `direction` (taux de réussite) ou `return` (moyenne des rendements).
- `threshold` / `topk` contrôlent la sélection des bins : seuil sur la proba ou top-k meilleurs profils.
- `combine` indique comment combiner plusieurs dimensions (`and`, `or`, `sum`).
- `by_session` active la dimension `session` (Asia, Europe, EU_US_overlap, US, Other) basée sur l'heure UTC.
- `by_month_start` et `by_month_end` ajoutent des flags booléens pour le premier et le dernier jour du mois.
- `by_news_hour` ajoute `is_news_hour` (heures macro sensibles 13h, 14h, 20h UTC).
- `by_third_friday` ajoute `is_third_friday` pour le 3ᵉ vendredi de chaque mois (expiration d'options).
- `by_rollover_day` expose `is_rollover_day` lorsque la série contient un `roll_id` (changement de contrat).
- `by_week_in_month` agrège par semaine dans le mois (`week_in_month` ∈ [1,5]) pour capturer les effets payroll/FOMC.
- `by_day_in_month` ajoute le bin exact du jour (`day_in_month`) et les tags `last_5`…`last_1` via `by_month_last_days`.
- `by_quarter` fournit `quarter` (1 à 4) pour mesurer les effets trimestriels.
- `by_month` ajoute désormais les dims `month` et `month_of_year` afin d'empiler plusieurs années.
- `by_month_flags` expose les indicateurs `is_january`…`is_december` pour isoler un mois précis.

Les colonnes `is_news_hour`, `is_third_friday` et `is_rollover_day` sont calculées automatiquement dans les features. Elles permettent d'isoler les heures clés des publications économiques, les séances d'expiration d'options mensuelles et les journées de rollover des contrats dérivés.

### Cycles intra-mois

Les features enrichies ajoutent `day_in_month`, `week_in_month` et les tags `last_5`…`last_1` pour identifier les cinq derniers jours ouvrés du mois. Activez-les via `by_day_in_month`, `by_week_in_month` et `by_month_last_days` afin de comparer, par exemple, la perf du 1ᵉʳ trading day vs. la fin de mois comptable.

### Saisons annuelles

Outre `month` / `month_of_year`, vous pouvez analyser `quarter` (1–4) et les flags `is_january`…`is_december`. Ces indicateurs permettent d'empiler plusieurs années et d'isoler des effets spécifiques (rallye de janvier, sell-in-may, clôtures trimestrielles, etc.).

### Exemple d'activation sessions & fins de mois

```json
{
  "profile": {
    "by_hour": false,
    "by_dow": false,
    "by_month": false,
    "by_session": true,
    "by_month_start": false,
    "by_month_end": true,
    "measure": "direction",
    "ret_horizon": 1,
    "min_samples_bin": 100
  },
  "signal": {
    "method": "threshold",
    "threshold": 0.55,
    "dims": ["session", "is_month_end"],
    "combine": "and"
  }
}
```

### Exemple `dims` intra-mois + trimestre

```json
{
  "profile": {
    "by_week_in_month": true,
    "by_quarter": true,
    "measure": "return",
    "ret_horizon": 4,
    "min_samples_bin": 50
  },
  "signal": {
    "method": "topk",
    "topk": 5,
    "dims": ["week_in_month", "quarter"],
    "combine": "sum"
  }
}
```

### Métriques conditionnelles stockées dans `seasonality_profiles.parquet`

| Colonne | Description |
| --- | --- |
| `run_len_up_mean` | Longueur moyenne des runs haussiers démarrant dans le bin. |
| `run_len_down_mean` | Longueur moyenne des runs baissiers démarrant dans le bin. |
| `n_runs` | Nombre de runs observés dans le bin. |
| `p_reversal_n` | Probabilité qu'un run se retourne en ≤ `ret_horizon` barres (estimateur Wilson). |
| `p_reversal_ci_low` / `p_reversal_ci_high` | Intervalle de confiance Wilson 95 % pour `p_reversal_n`. |
| `p_reversal_lift` | Écart du taux de reversal vs. le baseline du symbole. |
| `p_reversal_baseline` | Probabilité de reversal globale pour le symbole. |
| `amp_mean` | Amplitude moyenne (high-low) conditionnelle au bin. |
| `amp_std` | Écart-type de l'amplitude (high-low). |
| `amp_p25` / `amp_p50` / `amp_p75` / `amp_p90` | Quantiles conditionnels de l'amplitude high-low. |
| `atr_mean` | Moyenne de l'ATR si la série contient cette colonne. |
| `p_breakout_up` | Fréquence de franchissement du plus-haut de la veille. |
| `p_breakout_down` | Fréquence de cassure du plus-bas de la veille. |
| `p_in_range` | Probabilité de rester dans le range de la veille. |
| `ret_p25` / `ret_p50` / `ret_p75` / `ret_p90` | Quantiles conditionnels du rendement `ret_horizon`. |

## 📖 Documentation

- [Architecture & Récap Fonctionnel](docs/architecture_overview.md)
- [Market Stats – Notes & Garde-fous](docs/market_stats_guidelines.md)

