# Seasonality – Dimensions & Métriques (Référence)

Ce document résume toutes les **dimensions** temporelles et **métriques** de saisonnalité exposées par `seasonality/` (compute, profiles, runner) et accessibles via API/CLI.

> ⚠️ Remarque : certaines dimensions ou métriques peuvent apparaître après passage des prompts V2→V5. Cette page couvre l’ensemble cible.

---

## Dimensions temporelles supportées

| Dimension          | Type        | Valeurs (ex.)                     | Notes / Source |
|-------------------|-------------|-----------------------------------|----------------|
| `hour`            | catégoriel  | 0–23                              | dérivé de `ts` (UTC) |
| `dow`             | catégoriel  | 0–6 (Mon=0 … Sun=6)               | dérivé de `ts` (UTC) |
| `month` (`month_of_year`) | catégoriel | 1–12                       | dérivé de `ts` |
| `session`         | catégoriel  | `Asia`, `Europe`, `EU_US_overlap`, `US`, `Other` | heuristique horaire UTC |
| `is_month_start`  | bool        | 0/1                               | `ts.day == 1` |
| `is_month_end`    | bool        | 0/1                               | `ts.day == last_day(ts.month)` |
| `week_in_month`   | catégoriel  | 1–5                               | semaine calendaire dans le mois |
| `day_in_month`    | catégoriel  | 1–31 (+ éventuellement `last_5…last_1`) | granularité fine intra-mois |
| `quarter`         | catégoriel  | 1–4                               | dérivé de `ts` |
| `is_news_hour`    | bool        | 0/1                               | proxy heures macro (13,14,20 UTC) |
| `is_rollover_day` | bool        | 0/1                               | si `roll_id` change (futures) |
| `is_third_friday` | bool        | 0/1                               | expiration options (Opex) |

> **Bonnes pratiques** : normaliser `ts` en **UTC**, vérifier la **couverture** (pas de trous), et appliquer un **n_min** (taille minimale d’échantillon par bin).

---

## Métriques directionnelles & de rendement

| Métrique                      | Type     | Définition / Détails |
|------------------------------|----------|----------------------|
| `p_hat`                      | proba    | Proportion de hausses : `successes / n` où `success = (close_{t+h} > close_t)` |
| `ci_low`, `ci_high` (Wilson) | IC 95%   | Intervalle fréquentiste (Wilson) pour `p_hat` |
| `baseline`                   | proba    | Proba globale (tous bins confondus) pour le même symbole/target/split |
| `lift`                       | proba    | `p_hat - baseline` |
| `ret_mean`                   | moyenne  | Moyenne de `(close_{t+h}/close_t - 1)` |
| `ret_std`                    | std      | Écart-type des retours |
| `ret_q25/q50/q75/q90`        | quantiles| Quantiles de la distribution des retours (si activé) |

> `h` = `ret_horizon` (par défaut 1 barre). Les outcomes utilisent **t+1..t+h** (pas de lookahead).

---

## Métriques de volatilité / amplitude

| Métrique                | Type       | Définition / Détails |
|------------------------|------------|----------------------|
| `amp_mean`             | moyenne    | Moyenne de `high - low` (par barre) |
| `amp_std`              | std        | Écart-type d’amplitude |
| `amp_q25/q50/q75/q90`  | quantiles  | Distribution conditionnelle des amplitudes |
| `atr_mean`             | moyenne    | ATR moyen (si calculé côté features) |
| `baseline_amp` / `lift_amp` | comparatif | Moyenne/écart vs baseline globale |

---

## Séquences, reversals & breakouts

| Métrique                      | Type     | Définition / Détails |
|------------------------------|----------|----------------------|
| `run_len_up_mean`            | moyenne  | Longueur moyenne des séquences up (closes montants consécutifs) |
| `run_len_down_mean`          | moyenne  | Idem pour down |
| `p_reversal_n`               | proba    | Probabilité de retournement dans les `n` barres suivantes |
| `p_breakout_up`              | proba    | Proba que `high_t` casse le `prev_day_high` |
| `p_breakout_down`            | proba    | Proba que `low_t` casse le `prev_day_low` |
| `p_in_range`                 | proba    | Proba de rester dans `[prev_day_low, prev_day_high]` |

> Les probas sont accompagnées de **CI Wilson** et de **lift** vs baseline.

---

## Estimation & Robustesse (rappel)

- **Fréquentiste** : `p_hat` + **Wilson 95% CI**.  
- **Bayésien** (si activé dans `stats/`) : Beta–Binomial (prior Jeffreys), `p_mean`/`hdi`.  
- **Multiplicité** : FDR (Benjamini–Hochberg) si vous testez de nombreux bins.  
- **n_min** : ignorer ou marquer `insufficient=true` si `n < n_min`.  
- **WFA** : séparer `train/test` pour éviter l’overfit temporel ; apprendre les bins/tertiles sur **train**.

---

## Artefacts & API

- **Parquet** : `seasonality_profiles.parquet` (contient `symbol,timeframe,dim,bin,n,metrics...,start,end,spec_id,dataset_id`).  
- **API** :  
  - `POST /seasonality/run` (exécute un run saisonnalité)  
  - `POST /seasonality/optimize` (Optuna sur hyperparams saisonnalité)  
  - `GET /seasonality/profiles` (filtrage par `symbol,dim,metrics`)  
  - (Optionnel) `qe seasonality compare` pour corréler deux symboles sur une dimension.

---

## Exemples de Specs

### 1) Sessions + fin de mois (directionnelle)
```json
{
  "data": { "dataset_path": "data/eurusd_m1.csv", "symbols": ["EURUSD"], "timeframe": "M1", "start": "2025-01-01", "end": "2025-06-30" },
  "profile": { "by_hour": true, "by_dow": true, "measure": "direction", "ret_horizon": 1, "min_samples_bin": 300 },
  "signal":  { "method": "threshold", "threshold": 0.54, "dims": ["session","is_month_end"], "combine": "and" },
  "validation": { "scheme": "walk_forward", "train_months": 2, "test_months": 1, "folds": 3, "embargo_days": 2 },
  "artifacts": { "out_dir": "runs/seasonality_sessions_month_end" }
}

2) Amplitude & breakouts (analyse descriptive)

{
  "data": { "dataset_path": "data/btcusdt_m1.csv", "symbols": ["BTCUSDT"], "timeframe": "M1", "start": "2025-03-01", "end": "2025-06-30" },
  "profile": { "by_hour": true, "by_dow": true, "measure": "direction", "ret_horizon": 1, "min_samples_bin": 200 },
  "signal":  { "method": "topk", "topk": 3, "dims": ["hour"], "combine": "or" },
  "artifacts": { "out_dir": "runs/seasonality_btc_amplitude_breakouts" }
}


---

CLI rapides

# Lancer un run saisonnalité
poetry run qe seasonality run --spec specs/seasonality_eurusd_m1.json

# Optimiser les hyperparams saisonnalité (Optuna)
poetry run qe seasonality optimize --spec specs/seasonality_eurusd_m1.json

# Voir les meilleurs bins (selon métrique)
poetry run qe seasonality show --symbol EURUSD --dim session --metrics p_hat,amp_mean --top 10
