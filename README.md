![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)

# Quant Engine

## Présentation
Moteur d’optimisation et de backtest basé sur une spécification JSON, prenant en charge EMA/VWAP, TP/SL, Walk Forward Analysis, Optuna, MySQL et MLflow.

### Vue d’ensemble du moteur Python
- **Source de vérité métier** : le code Python produit les signaux (EMA/grid/DCA), reconstruit les trades, calcule les métriques de performance et garde la cohérence entre backtests et exécution live.
- **Séparation des rôles** :
  - *Signaux* (bas niveau, multi BUY par cycle),
  - *Trades agrégés* (1 cycle DCA = 1 trade logique, clôturé sur le SELL take‑profit),
  - *Run de stratégie* (métriques globales dérivées des trades).
- **Persistance** : Python ne stocke pas durablement les résultats métiers ; il calcule et transmet un payload vers le backend Java Spring responsable de l’import et de la conservation des runs/trades.

Affichage Markdown :  Ctrl + Shift + V
## Installation
```bash 
# Installer les dépendances nécessaires
pip install poetry 

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
- [Optimization workflow](docs/optimization.md)
- [Seasonality – Dimensions & Métriques](docs/seasonality_reference.md)
- [Live Trading Runner](docs/live.md)
- [High-level Strategies](docs/strategies_overview.md)

## Lancer l'API
```bash
poetry run uvicorn quant_engine.api.app:app --reload --app-dir src
```

## CLI
- **run-local**
  ```bash
  poetry run qe run-local --spec path/to/spec.json
  ```
  Utilise le job manager si disponible, sinon le runner legacy (`optimize.runner`).
- **submit**
  ```bash
  poetry run qe submit --spec path/to/spec.json
  ```
- **runs list**
  ```bash
  poetry run qe runs list --status running
  ```
- **runs show**
  ```bash
  poetry run qe runs show RUN_ID
  ```
- **backtest run**
  ```bash
  poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1.json
  ```
  Le backtest "signal" accepte `data.dataset_path` (CSV/JSON), `data.mysql`, ou les champs `data.delta_*` / `data.mysql_env` (Delta -> MySQL -> Java).
  Exemple Delta/MySQL : `specs/examples/backtest_eurusd_m1_delta_mysql.json`.
- **backtest optimize**
  ```bash
  poetry run qe backtest optimize --spec specs/examples/backtest_eurusd_m1_optimize.json
  ```
- **strategy optimize**
  ```bash
  poetry run qe strategy optimize --spec specs/strategy_dca_equity_example_minimal_optimize.json
  ```
  Ces commandes utilisent l'optimiseur officiel `optimize.variants` (workflow avancé).
- **export-delta-csv**
  ```bash
  poetry run qe export-delta-csv --asset-class crypto --symbol BTC --timeframe 1h
  ```
  Exporte un CSV OHLC vers `specs/examples/data/<asset>` au format `Timestamp,Open,High,Low,Close,Volume`.
  Pour `ACTION`/`ETF`, fournir `--delta-exchange` (ex: `NASDAQ`) car le path Delta inclut un segment exchange.

### Optimization spec (grid/random)
Define `optimization.search_space` with discrete lists or `{min,max,step}` ranges. Paths can target nested fields (use dots and list indexes).

```json
{
  "optimization": {
    "method": "grid",
    "objective": "sharpe",
    "search_space": {
      "signal.params.fast": {"min": 20, "max": 100, "step": 10},
      "strategy.params.grid[0].dd": [-10, -20, -30],
      "strategy.params.tp_sl.rules[0].tp_pct": {"min": 10, "max": 30, "step": 5}
    }
  }
}
```

### Composite objective + screening + promotion (exemple)
Voir aussi `docs/optimization.md` pour le workflow complet.

```json
{
  "optimization": {
    "objective": {
      "weights": {
        "sharpe": 1.0,
        "return_pct": 0.3,
        "max_drawdown_pct": -0.5
      }
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

## Performance, DCA et intégration backend
- **Module `quant_engine.performance`** : calcule les métriques côté Python et agrège les signaux DCA en `CompletedTrade` (1 cycle = 1 trade logique). Les BUY successifs d’un cycle sont consolidés ; la vente `take_profit` clôture le trade et porte les métadonnées du cycle.
- **StrategyRunResult** : résumé d’un run (dates, ratios win/loss, drawdown, Sharpe/Sortino calculés en Python). Les champs capital/prix/qty peuvent rester des placeholders selon la stratégie ; les valeurs optionnelles sont envoyées dans `extra` pour compatibilité future.
- **Flux de données** : `candles → signaux → trades → métriques → payload backend`. Les perfs sont calculées en Python pour garantir la cohérence entre backtest et live, éviter la duplication de logique et permettre la reproductibilité.
- **Payload Java** : un seul endpoint d’import est appelé avec `{ "run": {..}, "trades": [...] }`. Le backend Spring ne recalcule pas les perfs ; il persiste simplement le run et les trades reçus.
- **Conventions récentes** : `runId` est obligatoire et présent dans tous les objets envoyés. Les signaux sont des événements bas niveau ; les trades reflètent les cycles stratégiques ; le run agrège la performance globale. Certains champs prix/qty peuvent encore être complétés par la suite (TODO connus), mais la structure de payload est stable.

## Stress Tests & Monte Carlo
Les stress tests et Monte Carlo sont calculÃ©s cÃ´tÃ© Python et exposÃ©s dans le payload (puis persistÃ©s en DB).

- **Backtest** : Monte Carlo + scÃ©narios sont calculÃ©s sur lâ€™equity curve du backtest (PnL cumulÃ©).
- **DCA** : Monte Carlo niveau 1 est attachÃ© dans `run.extra["stress_tests"]` cÃ´tÃ© DCA.
- **Sortie** : le payload inclut `stress_tests` avec `monte_carlo` et `scenarios` (donnÃ©es brutes + paramÃ¨tres).

### Source Monte Carlo (equity / returns / trades)
Par defaut, le backtest calcule le Monte Carlo sur l'equity curve. Vous pouvez forcer la source :
- `source: "equity"` (par defaut)
- `source: "returns"` (PnL par trade)
- `source: "trades"` (trades + timestamps pour block bootstrap)

Exemple :
```json
{
  "performance": {
    "stress_tests": {
      "enabled": true,
      "monte_carlo": {
        "source": "trades",
        "n_sims": 300,
        "method": "bootstrap"
      }
    }
  }
}
```


#### Quand utiliser quelle source ?
- `equity` : par defaut. Recommande si tu veux refleter la dynamique globale de l'equity curve.
- `returns` : resample direct des PnL par trade. Recommande pour tester la robustesse au sequence de trades.
- `trades` : resample des trades avec timestamps (utile avec method=block/block_bootstrap).


### Sizing jitter (Monte Carlo)
Permet de simuler une incertitude de sizing (multiplicateur applique aux PnL par trade).

Champs supportes :
- `sizing.dist`: `uniform` (defaut), `normal`, `lognormal`
- `sizing.low` / `sizing.high` (uniform)
- `sizing.mu` / `sizing.sigma` (normal/lognormal)
- `sizing.min` / `sizing.max` (clamp optionnel)

Exemple :
```json
{
  "performance": {
    "stress_tests": {
      "enabled": true,
      "monte_carlo": {
        "n_sims": 300,
        "method": "bootstrap",
        "sizing": {
          "dist": "uniform",
          "low": 0.8,
          "high": 1.2
        }
      }
    }
  }
}
```

### TP/SL jitter (execution)
Permet de simuler une execution plus/moins favorable quand TP/SL est touche.
Le jitter est applique uniquement si un TP/SL est declenche (pas sur les exits par signal).

Champs supportes :
- `tpsl.jitter.dist`: `uniform` (defaut), `normal`
- `tpsl.jitter.tp_bps`: amplitude (bps) pour les sorties TP
- `tpsl.jitter.sl_bps`: amplitude (bps) pour les sorties SL
- `tpsl.jitter.seed`: seed optionnelle pour des runs deterministes

Exemple :
```json
{
  "tpsl": {
    "atr_window": 14,
    "atr_k": 1.0,
    "r_mult": 2.0,
    "jitter": {
      "dist": "uniform",
      "tp_bps": 5.0,
      "sl_bps": 8.0,
      "seed": 123
    }
  }
}
```
Exemple (dist normal) :
```json
{
  "tpsl": {
    "atr_window": 14,
    "atr_k": 1.0,
    "r_mult": 2.0,
    "jitter": {
      "dist": "normal",
      "tp_bps": 6.0,
      "sl_bps": 10.0,
      "seed": 321
    }
  }
}
```
Specs exemples :
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_tpsl_jitter.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_tpsl_jitter_light.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_tpsl_jitter_light_strict.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_tpsl_jitter_normal.json`

### Time distribution (durations / sessions)
Permet de perturber la distribution temporelle des trades (durations ou horaires) pour le calcul de `time_to_recovery`
et le CAGR. Requiert des timestamps (ex: `source: "trades"` ou returns/equity avec timestamps).

Champs supportes :
- `time_distribution.mode`: `exit_deltas` (defaut), `exit_times` (preserve distribution des horaires/sessions), `trade_durations`
- `time_distribution.seed`: seed optionnelle

Exemple :
```json
{
  "performance": {
    "stress_tests": {
      "monte_carlo": {
        "source": "trades",
        "time_distribution": {
          "mode": "exit_times",
          "seed": 321
        }
      }
    }
  }
}
```
Specs exemples :
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_light.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_light_strict.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_trade_durations.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_trade_durations_light.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_trade_durations_light_strict.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_exit_deltas.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_exit_deltas_light.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_exit_deltas_light_strict.json`

### Param drift (Monte Carlo)
Permet de simuler un drift des parametres de strategie en appliquant un multiplicateur sur les PnL.
Utile pour capturer une degradation progressive des parametres optimises.

Champs supportes :
- `param_drift.mode`: `per_sim`, `per_trade`, `random_walk`
- `param_drift.dist`: `normal` (defaut), `uniform`
- `param_drift.mu` / `param_drift.sigma` (normal)
- `param_drift.low` / `param_drift.high` (uniform)
- `param_drift.min` / `param_drift.max` (clamp optionnel)
- `param_drift.seed`: seed optionnelle

Exemple :
```json
{
  "performance": {
    "stress_tests": {
      "monte_carlo": {
        "source": "trades",
        "param_drift": {
          "mode": "random_walk",
          "dist": "normal",
          "mu": 1.0,
          "sigma": 0.05,
          "min": 0.7,
          "max": 1.3,
          "seed": 777
        }
      }
    }
  }
}
```
Specs exemples :
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_param_drift.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_param_drift_light.json`
- `specs/examples/backtest_eurusd_m1_csv_stress_10k_param_drift_light_strict.json`
### Mode "light" / "light_strict" (rÃ©duction taille)
Pour rÃ©duire la taille des rÃ©sultats, active un mode light sur Monte Carlo :
```json
{
  "performance": {
    "stress_tests": {
      "enabled": true,
      "monte_carlo": {
        "n_sims": 300,
        "output": {
          "mode": "light",
          "max_curves": 30,
          "curve_stride": 10
        }
      }
    }
  }
}
```
- `max_curves` limite le nombre de trajectoires conservÃ©es.
- `curve_stride` downsample les points des courbes.

Mode `light_strict` (percentiles enrichis + distributions minimales) :
```json
{
  "performance": {
    "stress_tests": {
      "enabled": true,
      "monte_carlo": {
        "n_sims": 300,
        "output": {
          "mode": "light_strict",
          "max_curves": 50,
          "curve_stride": 10
        }
      }
    }
  }
}
```
- `light_strict` conserve surtout les percentiles (p10/p25/p75/p90/p95/p99 + mean/std) et un Ã©chantillon de courbes.

### Optimisation Monte Carlo (NumPy)
Le Monte Carlo utilise une implÃ©mentation NumPy optimisÃ©e si NumPy est disponible. Le fallback Python reste actif si besoin.

### Persistance MySQL (Option A)
Les stress tests sont persistÃ©s dans la table `stress_test_result` (JSON brut).
SchÃ©ma alignÃ© sur le backend Java :
```sql
CREATE TABLE IF NOT EXISTS stress_test_result (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  strategy_id VARCHAR(128),
  run_id VARCHAR(64) NOT NULL,
  asset_class VARCHAR(32),
  symbol VARCHAR(64),
  timeframe VARCHAR(32),
  mode VARCHAR(32) NOT NULL,
  payload_json JSON NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Si la colonne `payload_json` est trop petite (erreur MySQL 1406), Python tente automatiquement une version compacte.
Tu peux forcer une taille max avec :
```
QE_STRESS_TEST_MAX_JSON_CHARS=60000
```

### High-level Strategies

- **Backtest**
  ```bash
  poetry run qe strategy backtest --spec specs/strategy_dca_equity_example.json
- **Live (avec implémentation)**
  ```bash
  poetry run qe live run --spec specs/live_example.json
  Assurez-vous que `specs/live_example.json` contient `"strategy": { "impl": { "type": "dca_equity", ... } }`.

## Exemples de filtres

## Exemples d'optimization
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

### Volatilité & tendance

```bash
poetry run qe stats run --spec specs/filters_volatility_trend_example.json
```

> Consulte la [référence des filtres](docs/filters.md) pour le détail des paramètres ADX/ATR/EMA slope et des autres filtres disponibles.

### Structure & ICT

```bash
poetry run qe stats run --spec specs/filters_structure_ict_example.json
```

### Seasonality & Time

```bash
poetry run qe stats run --spec specs/filters_time_seasonality_example.json
```

### Statistical & Probabilistic

```bash
poetry run qe stats run --spec specs/filters_stat_prob_example.json
```

### Risk & Money Management

```bash
poetry run qe stats run --spec specs/filters_risk_mgmt_example.json
```

## Configuration `.env`
poetry run pip install deltalake
poetry run python -m pip install deltalake
python -m pip install typer
python -m pip install pandas
python -m pip install pydantic
python -m pip install sqlalchemy
export DB_DSN="mysql+pymysql://restadmin:ronanronan77@127.0.0.1:3306/restdb"
poetry run pip install pandas-market-calendars
poetry add httpx
poetry run qe strategy backtest --spec specs/strategy_dca_equity_example_minimal.json
poetry run qe strategy backtest --spec specs/strategy_dca_equity_example_minimal_with_filters.json
poetry run qe strategy backtest --spec specs/strategy_dca_equity_example_minimal_with_filters_ema.json
poetry run qe strategy backtest --spec specs/strategy_dca_equity_example_minimal_stats_gate.json
poetry run qe strategy backtest --spec specs/strategy_dca_crypto_example_minimal.json
poetry run qe strategy backtest --spec specs/strategy_dca_crypto_example_minimal_with_filters.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_with_filters.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_with_filters_csv.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_stats_gate.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_delta_mysql.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_delta_mysql_with_filters.json
poetry run qe backtest optimize --spec specs/examples/backtest_eurusd_m1_optimize.json
poetry run qe strategy optimize --spec specs/strategy_dca_equity_example_minimal_optimize.json
poetry run qe strategy backtest --spec specs/examples/strategy_dca_etf_delta_2024_2026.json


## Ici limité temporairement à QE_STRESS_TEST_MAX_JSON_CHARS =60000 dans le code si on veut changer on peut set la variable 
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_delta_mysql_stress.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_light.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_light_strict.json

poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_light.json

poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_light_strict.json

poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_returns.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_returns_light.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_returns_light_strict.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_trades.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_trades_light.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_trades_light_strict.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_sizing.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_sizing_light.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_sizing_light_strict.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_tpsl_jitter.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_tpsl_jitter_light.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_tpsl_jitter_light_strict.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_tpsl_jitter_normal.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_light.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_light_strict.json

poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_trade_durations.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_exit_deltas.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_trade_durations_light.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_trade_durations_light_strict.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_exit_deltas_light.json
poetry run qe backtest run --spec specs/examples/backtest_eurusd_m1_csv_stress_10k_time_dist_exit_deltas_light_strict.json

poetry run qe backtest run specs/examples/backtest_eurusd_m1_csv_stress_10k_param_drift.json
poetry run qe backtest run specs/examples/backtest_eurusd_m1_csv_stress_10k_param_drift_light.json
poetry run qe backtest run specs/examples/backtest_eurusd_m1_csv_stress_10k_param_drift_light_strict.json

poetry run pytest tests/test_backtest_data_sources.py
poetry run pytest tests/test_strategy_dca_variants.py
poetry run pytest tests/test_backtest_trade_expectations.py
poetry run pytest tests/test_trade_expectations_details.py
poetry run pytest tests/test_optimize_variants_baseline.py
poetry run pytest tests/test_optimize_variants_filters.py
poetry run pytest tests/test_optimize_variants_screening.py
poetry run pytest tests/test_strategy_dca_stop_loss.py
poetry run pytest tests/test_stats_basic_spec.py
poetry run pytest tests/test_seasonality_basic_spec.py
poetry run pytest tests/test_stats_seasonality_combo.py
poetry run pytest tests/test_stats_gate_filter.py
poetry run pytest tests/test_combo_backtest_dca_seasonality.py
poetry run pytest tests/test_timeframe_variants.py
poetry run pytest tests/test_data_edge_cases.py
poetry run pytest -m slow tests/test_large_dataset_perf.py

## STRESS TESTS & MONTE CARLO
poetry run pytest tests/test_stress_tests.py
poetry run pytest tests/test_dca_stress_tests.py 
poetry run pytest tests/test_stress_tests_large_dataset.py
poetry run pytest tests/test_backtest_stress_sources.py
poetry run pytest tests/test_backtest_engine_tpsl_jitter.py
poetry run pytest tests/test_stress_tests_time_distribution.py
poetry run pytest tests/test_stress_tests_param_drift.py

## Ajouts de tests filters 
poetry run pytest tests/test_filters_mtf_anomaly.py
poetry run pytest tests/test_filters_benford_extended.py
poetry run pytest tests/test_backtest_filter_specs_extra.py
poetry run pytest tests/test_filters_market_manipulation.py
poetry run pytest tests/test_filters_htf_poi.py
poetry run pytest tests/test_filters_orderflow_delta.py
poetry run pytest tests/test_filters_macro_cot_oi.py
poetry run pytest tests/test_filters_lower_timeframe_confluence.py
poetry run pytest tests/test_filters_psychologic_news.py
poetry run pytest tests/test_filters_stationarity_full.py
poetry run pytest tests/test_filters_volatility_extras.py
poetry run pytest tests/test_filters_trade_filter_service.py
poetry run pytest tests/test_backtest_filter_rules_scoring.py
poetry run pytest tests/test_filters_ict_poi.py

## Utile pour des traces de perf plus rigoureuses
set QE_PERF_TRACE=1
## Pour voir avec des logs -s (log de perf générales mais pas détaillées comme avec QE_PERF_TRACE, faut avoir les deux)
poetry run pytest -m slow -s tests/test_large_dataset_perf.py
poetry run pytest -m slow -s tests/test_large_dataset_optimize_perf.py

poetry run pytest tests/test_strategy_dca_variants.py tests/test_backtest_trade_expectations.py tests/test_backtest_data_sources.py tests/test_optimize_variants_baseline.py




IL FAUT QUE LES VARIABLES D'ENV SOIT DECLAREES DANS LE TERMINAL, LE PROJET PREND PAS LE .ENV
Copier `.env.example` vers `.env` et ajuster :
```env
DB_DSN=sqlite:///.db/quant.db
DB_ECHO=false
MLFLOW_TRACKING_URI=http://localhost:5000
```
Pour les tests rapides, la persistance locale utilise SQLite (`.db/quant.db`). Configure les accès MySQL via `QE_MARKETDATA_MYSQL_URL` lorsque tu veux lire les OHLCV depuis ton instance Spring.


## Activer les alertes Telegram
Configure les variables d'environnement suivantes avant de lancer le runner live :

```bash
export ENABLE_TELEGRAM_ALERTS=true
export TELEGRAM_BOT_TOKEN="123456789:ABCDEF..."
export TELEGRAM_CHAT_ID="123456789"
```

Mets `ENABLE_TELEGRAM_ALERTS` à `false` (ou supprime la variable) pour désactiver les notifications. Si les variables sont absentes ou incomplètes, le moteur continue de fonctionner sans envoyer d'alerte et logue un avertissement.


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
- `POST /submit/async`
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

#### Statistiques (SQLite persisté)
```bash
curl.exe -X POST http://127.0.0.1:8000/stats/run -H "Content-Type: application/json" --data-binary @specs/examples/stats_run.json
# -> {"status":"completed","id":"JOB_ID"}
```

```bash
curl.exe http://127.0.0.1:8000/status/JOB_ID
curl.exe http://127.0.0.1:8000/result/JOB_ID | python -m json.tool
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

#### Mode async (jobs persistés)
Les endpoints suffixés par `/async` (`/submit/async`, `/stats/run/async`, `/levels/build/async`, `/levels/fill/async`,
`/seasonality/run/async`, `/seasonality/optimize/async`) mettent les jobs en file d'attente (statut `pending`)
et retournent immédiatement un `job_id`. Un worker peut ensuite dépiler les jobs et exécuter `run_next_job` pour
traiter la file.

```bash
curl.exe -X POST http://127.0.0.1:8000/submit/async -H "Content-Type: application/json" --data-binary @specs/examples/submit_local.json
# -> {"status":"pending","id":"JOB_ID"}
```

```bash
curl.exe http://127.0.0.1:8000/status/JOB_ID
curl.exe http://127.0.0.1:8000/result/JOB_ID | python -m json.tool
```

```bash
python - <<'PY'
from quant_engine.api import app

job_result = app.run_next_job()
print(job_result)
PY
```

#### Worker / Queue / Lifecycle
- `POST /runs` enfile un run canonique avec statut `QUEUED` (traitement async).
- La queue canonical (`api_jobs`) est stockee en SQLite dans cette implementation locale.
- API et worker doivent pointer vers le meme fichier SQLite (`DB_SQLITE_PATH`) sinon les runs restent en `QUEUED`.
- Le worker est d??di?? et se lance s??par??ment :

```bash
poetry run qe worker
# ou une it??ration unique (debug)
poetry run qe worker --once
```

Exemple recommande (meme variable dans les 2 terminaux):

```powershell
$env:DB_SQLITE_PATH="C:\Users\ronan\Desktop\Quant-Engine-Python\Quant-Python-Engine\.db\quant.db"
```

- Cycle de vie attendu: `QUEUED` -> `RUNNING` -> `SUCCEEDED | FAILED | CANCELED`.
- Cancel (best-effort) :

```bash
curl.exe -X POST http://127.0.0.1:8000/runs/RUN_ID/cancel
```

- Retries / timeouts configurables via env :
`QE_CANONICAL_MAX_ATTEMPTS` (defaut: 3)
`QE_CANONICAL_TIMEOUT_SECONDS` (timeout par job, optionnel)
`QE_CANONICAL_STALE_SECONDS` (reprise des jobs RUNNING stale)
`QE_WORKER_POLL_SECONDS` (polling du worker)

Endpoints lifecycle (canonical runs):
`GET /runs/{id}` -> { run_id, status, created_at, updated_at, ... }
`GET /runs/{id}/result` -> terminal: { run_id, status, result?, error? }
`GET /runs/{id}/result` -> non terminal: { run_id, status, message }
`POST /runs/{id}/cancel` -> 200 idempotent, 409 si d??j?? terminal
`GET /runs/capabilities?spec_type=dca` -> runtime capabilities matrix (supported vs accepted-but-not-wired)

Canonical DCA source of truth (support matrix + payload rules):
`docs/canonical_runs_dca_source_of_truth_2026-02-20.md`

Delta troubleshooting (`Delta source skipped ... base_uri_present=False`):
- Provide `data.delta_base` in spec or export `DELTA_BASE_URI`.
- Optional env (especially for crypto):

```powershell
$env:DELTA_BASE_URI="s3://quant-delta-dev"
$env:DELTA_PREFIX="delta"
$env:DELTA_QUOTES="USDT,USD"
$env:DELTA_MARKET_TYPE="SPOT"
$env:DELTA_EXCHANGE="BINANCE"
```

Capabilities example:

```bash
curl.exe "http://127.0.0.1:8000/runs/capabilities?spec_type=dca" | python -m json.tool
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
| Saisonalite (`/seasonality/*`) | bloc Saisonalite (`/seasonality/run`, `/seasonality/optimize`, `/seasonality/runs`, `/seasonality/profiles`) | `DB_DSN=sqlite:///.db/quant.db` ; `QE_MARKETDATA_MYSQL_URL` si lecture MySQL ; `polars` (optionnel) | Persistance dans `.db/quant.db` (`seasonality_*`) + artefacts `runs/`. Fallback pandas en mode lite si `polars` manque. `seasonality_optimize` s'appuie sur Optuna. |

Pense a exporter `DB_DSN=sqlite:///.db/quant.db` avant de lancer l'API, puis `QE_MARKETDATA_MYSQL_URL` vers `mysql+pymysql://restadmin:ronanronan77@127.0.0.1:3306/restdb?charset=utf8mb4` si tu veux lire tes OHLCV MySQL.

#### Nettoyage des artefacts
- Les API statistiques et saisonnalité écrivent dans `.db/quant.db`. Supprimer ce fichier pour repartir de zéro (`rm .db/quant.db` ou `Remove-Item .db/quant.db`).
- Les artefacts locaux sont générés dans `runs/` et les fichiers `summary.json` / `trials.parquet` à la racine. Supprimer ces éléments si nécessaire.
- `seasonality_run` et `seasonality_optimize` utilisent `polars` pour les profils complets ; sans `polars`, un mode lite pandas évite le crash mais produit des métriques réduites.

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
poetry run qe stats run --spec path/to/stats_spec.json
poetry run qe stats show --symbol EURUSD --event k_consecutive --target up_next_bar --timeframe M1 --limit 20
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
