# Quant Engine – Feature Catalog

## Market Stats
- **Événements** :
  - `k_consecutive` – détecte `k` bougies consécutives dans une direction donnée.【F:src/quant_engine/stats/events.py†L7-L24】
  - `shock_atr` – marque les bougies dont la True Range dépasse un multiple d’ATR.【F:src/quant_engine/stats/events.py†L26-L37】
  - `breakout_hhll` – signale les cassures de plus hauts/plus bas sur un lookback configurable.【F:src/quant_engine/stats/events.py†L38-L55】
- **Conditions** : `htf_trend` (EMA sur timeframe supérieur), `vol_tertile` (binning ATR), `session` (mapping horaire).【F:src/quant_engine/stats/conditions.py†L8-L46】
- **Targets** : `up_next_bar`, `continuation_n` (run-length directionnel), `time_to_reversal` (horizon max avant retournement).【F:src/quant_engine/stats/targets.py†L10-L35】
- **Estimateurs & contrôles** :
  - Fréquentiste : `freq_with_wilson` (p̂ + Wilson CI), `p_value_binomial_onesided_normal` pour les p-values.【F:src/quant_engine/stats/estimators.py†L19-L86】
  - Bayésien : posterior Beta-Binomial (`aggregate_binary_bayes`, `posterior_mean`, `beta_hdi`).【F:src/quant_engine/stats/estimators.py†L118-L178】
  - Multiplicité : correction Benjamini–Hochberg (`benjamini_hochberg`).【F:src/quant_engine/stats/estimators.py†L96-L114】
- **Production & persistance** : pipeline `runner.py` agrège p_hat, lifts (freq/bayes), flags `insufficient`, écrit en DB + artefacts Parquet.【F:src/quant_engine/stats/runner.py†L1-L160】【F:src/quant_engine/persistence/db.py†L120-L167】

## Seasonality
- **Dimensions couvertes** : hour, dow, month, session, flags calendaires (month_start/end, week_in_month, day_in_month, quarter, third_friday, etc.).【F:docs/seasonality_reference.md†L9-L48】
- **Métriques directionnelles & de rendement** : `p_hat`, `ci_low/ci_high`, `baseline`, `lift`, `ret_mean/std`, quantiles de retours.【F:docs/seasonality_reference.md†L52-L75】
- **Amplitude & séquences** : `amp_mean/std`, quantiles d’amplitude, probabilités de breakouts/reversal, run lengths.【F:docs/seasonality_reference.md†L77-L139】
- **Calcul** : `compute.prepare_features/compute_profiles` construit les tables conditionnelles et métriques dérivées (lift, metrics conditionnels).【F:src/quant_engine/seasonality/compute.py†L1-L80】【F:src/quant_engine/seasonality/compute.py†L82-L140】
- **Profils & règles** : `profiles.select_bins` filtre les bins selon seuil/top-k, `SeasonalityRules` encapsule les combinaisons (combine=and/or).【F:src/quant_engine/seasonality/profiles.py†L1-L112】
- **Runner & artefacts** : `runner.py` orchestre chargement dataset, Walk-Forward (`splitter.generate_folds`), sélection des bins, export DB (`seasonality_profiles`, `seasonality_runs`) et artefacts (`profiles.parquet`, `summary.json`, `trades/equity`).【F:src/quant_engine/seasonality/runner.py†L1-L120】【F:src/quant_engine/seasonality/runner.py†L120-L220】【F:src/quant_engine/persistence/db.py†L168-L215】
- **Optimisation Optuna** : `seasonality/optimize.py` normalise l’espace de recherche (dims, threshold/topk, min_samples), propose des suggestions, retourne `best_value/best_params`.【F:src/quant_engine/seasonality/optimize.py†L1-L90】【F:src/quant_engine/api/app.py†L690-L698】

## Backtest & TP-SL
- **Backtest** : moteur bar-based appliquant signaux EMA/VWAP, exécution trade-by-trade, agrège les métriques Sharpe/Sortino/MaxDD/CAGR/Hit rate/Avg-R.【F:docs/architecture_overview.md†L19-L43】【F:src/quant_engine/backtest/metrics.py†L1-L56】
- **TP/SL** : stops ATR multiples (`StopInitializer.fixed_atr`), take-profit en R multiples (`TakeProfit.r_multiple`).【F:docs/architecture_overview.md†L24-L33】【F:src/quant_engine/tpsl/rules.py†L1-L32】
- **Walk-Forward Analysis** : génération de folds train/test avec embargo via `splitter.generate_folds`.【F:docs/architecture_overview.md†L21-L29】【F:src/quant_engine/validate/splitter.py†L1-L38】
- **Optimisation** : runner Optuna explore search space EMA & R multiples, écrit artefacts `trials.parquet`, `summary.json`, `trades_*.parquet`, `equity_*.parquet`.【F:src/quant_engine/optimize/runner.py†L1-L76】【F:src/quant_engine/optimize/runner.py†L78-L120】

## Persistance & Artefacts
- **Base SQL (quant)** : tables `experiment_runs`, `run_metrics`, `trials`, `market_stats`, `seasonality_profiles`, `seasonality_runs` (clé unique, timestamps).【F:src/quant_engine/persistence/db.py†L48-L215】
- **Modèles dataclass** : `MarketStat`, `SeasonalityProfile`, `SeasonalityRun` pour manipuler les entités persistées.【F:src/quant_engine/persistence/models.py†L1-L58】
- **Formats disques** : `artifacts.write_*` produit `trials.parquet`, `equity_{fold}.parquet`, `trades_{fold}.parquet`, `summary.json`.【F:src/quant_engine/optimize/runner.py†L80-L118】
- **Sources marché** : lecture OHLCV via CSV/JSON (spec), ou MySQL via `load_ohlcv_mysql` alimentée par `QE_MARKETDATA_MYSQL_URL`.【F:README.md†L74-L123】【F:src/quant_engine/datafeeds/mysql_feed.py†L45-L118】
- **Cibles MySQL** : résultats écrits dans la base quant (MySQL/SQLite via SQLAlchemy/Alembic).【F:docs/architecture_overview.md†L11-L36】【F:src/quant_engine/persistence/db.py†L8-L215】
