![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)

# Quant Engine

## Présentation
Moteur d’optimisation et de backtest basé sur une spécification JSON, prenant en charge EMA/VWAP, TP/SL, Walk Forward Analysis, Optuna, MySQL et MLflow.

## Installation
```bash
poetry install
```

## Lancer l'API
```bash
poetry run uvicorn quant_engine.api.app:app --reload
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
DB_DSN=mysql+pymysql://quant:quant@localhost:3306/quant?charset=utf8mb4
DB_ECHO=false
MLFLOW_TRACKING_URI=http://localhost:5000
```

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

### CLI

```bash
poetry run quant-engine stats run --spec path/to/stats_spec.json
poetry run quant-engine stats show --symbol EURUSD --event k_consecutive --target up_next_bar --timeframe M1 --limit 20
```

### API

- `POST /stats/run`
- `GET /stats/result`
- `GET /stats`

## Seasonality Backtest

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
poetry run qe seasonality run --spec specs/eurusd_m1_seasonality.json
```

### Dimensions & signal

- `by_hour`, `by_dow`, `by_month` activent les agrégations par heure, jour de semaine ou mois pour les profils.
- `measure` choisit la métrique : `direction` (taux de réussite) ou `return` (moyenne des rendements).
- `threshold` / `topk` contrôlent la sélection des bins : seuil sur la proba ou top-k meilleurs profils.
- `combine` indique comment combiner plusieurs dimensions (`and`, `or`, `sum`).
- `by_session` active la dimension `session` (Asia, Europe, EU_US_overlap, US, Other) basée sur l'heure UTC.
- `by_month_start` et `by_month_end` ajoutent des flags booléens pour le premier et le dernier jour du mois.

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

## 📖 Documentation

- [Architecture & Récap Fonctionnel](docs/architecture_overview.md)
- [Market Stats – Notes & Garde-fous](docs/market_stats_guidelines.md)

