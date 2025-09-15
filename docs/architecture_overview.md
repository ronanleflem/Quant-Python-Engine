# Architecture & Récap Fonctionnel

Ce document résume l’état **actuel** du moteur lourd Python (après intégration des prompts backtest/optimisation + persistance + Market Stats).

## 1) Vue d’ensemble

- **API (FastAPI)** : soumettre des expériences (Spec JSON), suivre l’état, récupérer les résultats (backtests & stats).
- **CLI (Typer)** : exécutions locales et consultation (`qe run-local`, `qe submit`, `qe runs list/show`, `qe stats run/show`).
- **Core Backtest/Optimize** : moteur bar-based vectorisé, Walk-Forward Analysis (WFA), optimisation Optuna.
- **Market Stats** : calcul de probabilités conditionnelles (patterns simples) + intervalles de confiance + contrôle de multiplicité.
- **Persistence** : SQLAlchemy/Alembic (MySQL cible, SQLite fallback), artefacts Parquet/JSON, logging MLflow (si configuré).
- **Qualité & CI** : pytest, ruff, black, mypy, pré-commit, GitHub Actions.

Arborescence (simplifiée) :

src/quant_engine/ api/ (FastAPI, schémas Pydantic) cli/ (Typer CLI) core/ (spec, dataset, features) backtest/ (engine, metrics) tpsl/ (règles TP/SL) validate/ (splitter WFA) optimize/ (Optuna runner) stats/ (events, conditions, targets, estimators, runner) persistence/ (db, models, repo, migrations) io/ (artifacts, ids)

## 2) Backtest & Optimisation

- **Entrée** : `ExperimentSpec` (JSON) décrivant data, stratégie, filtres (EMA/VWAP…), exécution, TP/SL, validation (WFA), objectif, search space.
- **Moteur** : backtest bar-based (entrées/sorties au bar suivant), frais/slippage bps, position 1x (MVP).
- **TP/SL** : Stop ATR * k, Take Profit en R-multiples {1..5}.
- **Validation** : **Walk-Forward Analysis** (ex. 2 mois train / 1 mois test, embargo).
- **Optimisation** : **Optuna (TPE)** mono-objectif (ex. Sharpe), contrainte min_trades.
- **Sorties** : `trials.parquet`, `equity_{fold}.parquet`, `trades_{fold}.parquet`, `summary.json`, best params/metrics.

**Intérêt math/stat :**
- **WFA** limite le sur-apprentissage (test hors-échantillon).
- **Optuna** explore l’espace des hyperparamètres (EMA periods, ATR mult, etc.).
- **Métriques** : Sharpe, Sortino, MaxDD, CAGR, #trades, hit rate, avg-R.

## 3) Market Stats (contexte probabiliste)

But : mesurer des **régularités empiriques** utiles pour contextualiser l’action (go/no-go, moduler le sizing), indépendamment des stratégies.

### a) Éléments

- **Events** (bool) : `k_consecutive` (k bougies up/down), `shock_atr` (TR > mult*ATR), `breakout_hhll` (cassure HH/LL).
- **Conditions** (catégorielles) : `htf_trend` (EMA sur HTF), `vol_tertile` (ATR en tertiles), `session`.
- **Targets** : `up_next_bar`, `continuation_n`, `time_to_reversal`.

### b) Estimateurs & incertitudes

Deux approches calculées en parallèle :

- **Fréquentiste (Wilson 95% CI)**  
  - p̂ = succès / n  
  - Intervalle **Wilson** : plus stable que une approximation naïve.
- **Bayésien (Beta–Binomial, prior de Jeffreys)**  
  - Posterior Beta(succès+0.5, échecs+0.5)  
  - Moyenne postérieure, MAP, **HDI 95%** (intervalle de densité la plus crédible).

**Baselines & lifts** :  
- Baseline (globale par `symbol/target/split`) → `lift = p_est - baseline`.

### c) Multiplicité (contrôle FDR)

- p-values ≈ test proportion vs baseline (approx. normale + correction de continuité).
- **Benjamini–Hochberg** → q-values, flag `significant`.
- Évite les faux positifs quand plusieurs patterns sont testés.

### d) WFA pour les stats

- Splits **train/test** appliqués aux agrégats.
- (Recommandation) Binning (ex. tertiles de vol) appris sur **train**, appliqué sur test.

### e) Sorties & API

- Table DB `market_stats` :  
  `symbol, timeframe, event, condition_name, condition_value, target, split, n, successes, p_hat, ci_low, ci_high, p_mean, p_map, hdi_low, hdi_high, lift_freq, lift_bayes, p_value, q_value, significant, start, end, spec_id, dataset_id, created_at`
- Parquet `stats_summary.parquet`.
- Endpoints : `/stats`, `/stats/summary`, `/stats/heatmap`, `/stats/top`.
- CLI : `qe stats run`, `qe stats show`.

**Usage pratique** :  
Ex. “Avec 2 bougies up en M1, HTF up, vol haute → P(up_next) ≈ 57% [54–60%]” → éviter un short contre le contexte.

## 4) Persistence & Observabilité

- **SQLAlchemy + Alembic** : MySQL (prod/dev) ou SQLite fallback (tests, CI).
- **MLflow** (si configuré) : tracking runs/params/metrics/artefacts.
- **Artefacts** sur disque : Parquet/JSON (equity, trades, trials, summary).
- **API lecture** : `/runs`, `/runs/{id}`, `/runs/{id}/trials`, `/runs/{id}/metrics` (+ endpoints stats).

## 5) Outils & Bibliothèques clés

- **FastAPI**, **Typer** – API & CLI  
- **Pydantic** – validation des Specs  
- **Optuna** – optimisation d’hyperparamètres  
- **SQLAlchemy**, **Alembic** – persistance et migrations  
- **Polars/Numpy** – calculs vectorisés  
- **Pytest**, **Ruff**, **Black**, **MyPy**, **pre-commit** – qualité & tests  
- **GitHub Actions** – CI (lint, type-check, tests)

## 6) Bonnes pratiques (rappel)

- **No lookahead** : conditions calculables à t ; targets sur t+1..t+n.  
- **n_min** : seuil d’échantillon (ex. 300) ; sous le seuil → `insufficient=true`.  
- **WFA** : séparer train/test ; apprendre les bins sur train.  
- **Contrôle FDR** : BH sur p-values si beaucoup de patterns.  
- **Traçabilité** : stocker `spec_id`, `dataset_id`, dates `start/end` dans chaque sortie.

---

> Ce document reflète l’état courant du moteur. Pour les détails d’implémentation : voir les fichiers sous `src/quant_engine/` et la documentation du README.
