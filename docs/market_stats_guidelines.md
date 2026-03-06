# Market Stats – Notes & Garde-fous

Ce module calcule des statistiques conditionnelles de marché (probabilités simples, run-lengths, continuation, reversion, etc.) utiles comme **contexte** pour les stratégies.

## Principes de rigueur

- **Définitions figées**  
  - Exemple : "Up candle = close > open"  
  - Exemple : "tf_multiplier=60 ⇒ HTF = 60 × timeframe de base"

- **No lookahead**  
  - Les conditions doivent être calculables à t.  
  - Les cibles (targets) utilisent uniquement t+1..t+n.

- **Échantillon minimum**  
  - n_min (par défaut 300).  
  - Si n < n_min ⇒ résultat marqué `insufficient=true`.

- **WFA (walk-forward analysis)**  
  - Splits train/test.  
  - Les bins (ex. tertiles ATR) doivent être définis sur **train** et appliqués sur test.

- **Multiplicité**  
  - Plusieurs patterns ⇒ risque de faux positifs.  
  - Utiliser un contrôle de FDR (Benjamini–Hochberg).

- **Intervalles de confiance**  
  - Fréquentiste : Wilson 95% CI.  
  - Bayésien : Beta-Binomial (Jeffreys prior) + HDI 95%.

- **Binning fixe**  
  - Pas de re-binning par split test.  
  - Exemple : definir les tertiles de volatilite sur train uniquement.

- **Stats conditionnelles**  
  - Limiter le nombre de dimensions (ex. session ou day-of-week).  
  - Garder un fallback global si n < n_min.  
  - Conditions pretes: `hour_bin`, `day_of_week`, `month_of_year`, `session_from_ts`.

- **Rolling recalibration**  
  - Rafraîchir les stats sur fenêtres glissantes (ex. 6 mois).  
  - Monitorer le drift temporel des probabilités.

## Pourquoi ces garde-fous ?

- Éviter l’auto-intox (overfit sur un dataset).  
- Savoir si un pattern est **réellement robuste** ou juste du bruit.  
- Pouvoir comparer différentes conditions de marché de manière saine.

---

👉 À lire **avant** d’ajouter de nouveaux events/conditions/targets.


## Candle structure coverage (already implemented)

Les patterns "candle structure" ne necessitent pas un nouveau filtre OHLC. Ils
existent deja comme events/targets dans les market stats:

- Events (src/quant_engine/stats/events.py):
  - bullish_engulfing, bearish_engulfing
  - bullish_streak, bearish_streak (k param)
  - gap_up, gap_down
  - bullish_candle, bearish_candle
- Targets (src/quant_engine/stats/targets.py):
  - next_bullish, next_bearish (ex: "Bullish -> Bullish")
  - body_ratio, upper_wick_ratio, lower_wick_ratio
  - candle_std, candle_zscore
  - breakout_high_first, breakout_low_first
  - retracement_probability

Si tu veux filtrer ces patterns en runtime, utilise un stats run + stats_gate
(event/target/metric) au lieu de recoder un filtre OHLC.


## Example: stats spec + stats_gate for candle patterns

## Canonical `stats_pack` guidance

Le contrat canonical `POST /runs` supporte maintenant un mode `data.stats_pack`
pour lancer des calculs enrichis sans imposer un seul triplet
`stats.event/stats.condition/stats.target`.

Packs exposes par le runtime Python:

- `candle_structure`
- `volatility_shocks`
- `gaps_breakouts`
- `all_basic` (union deterministe des trois packs)

Regles:

- Un pack etend uniquement `events[]` et `targets[]`.
- Si `data.stats_pack` et le triplet mono (`stats.event` / `stats.target`) sont fournis ensemble,
  le runtime fait une union deterministe; le triplet n'ecrase pas le pack.
- `stats.condition` reste optionnel et, s'il est fourni, s'applique a tout le pack.
- L'absence de `stats.condition` produit des stats globales (`condition_name` /
  `condition_value` nuls) sans lookahead additionnel.
- Les metriques enrichies calculees par le runner sont persistees dans
  `market_stats`: `p_mean`, `p_map`, `hdi_low`, `hdi_high`, `lift_freq`,
  `lift_bayes`, `p_value`, `q_value`, `significant`, `insufficient`.

Example StatsSpec (events + targets) to compute candle structure metrics:

```json
{
  "data": {
    "dataset_path": "tests/data/ohlcv.csv",
    "symbols": ["EURUSD"],
    "timeframe": "1m",
    "start": "2025-01-01",
    "end": "2025-01-02"
  },
  "events": [
    {"name": "bullish_engulfing"},
    {"name": "bearish_engulfing"},
    {"name": "gap_up"},
    {"name": "gap_down"},
    {"name": "bullish_streak", "params": {"k": 3}}
  ],
  "targets": [
    {"name": "next_bullish"},
    {"name": "next_bearish"},
    {"name": "body_ratio"},
    {"name": "upper_wick_ratio"},
    {"name": "lower_wick_ratio"}
  ],
  "persistence": {
    "enabled": true,
    "dataset_id": "EURUSD_2025_01",
    "spec_id": "candle_struct_v1"
  }
}
```

Example stats_gate filter using those persisted stats:

```json
{
  "filters": [
    {
      "type": "stats_gate",
      "params": {
        "event": "bullish_engulfing",
        "target": "next_bullish",
        "metric": "p_hat",
        "comparator": ">=",
        "threshold": 0.55,
        "symbol": "EURUSD",
        "timeframe": "1m",
        "allow_if_missing": false,
        "allow_if_insufficient": false
      }
    }
  ]
}
```


### Example: stats_gate with condition (session) + min_samples

```json
{
  "filters": [
    {
      "type": "stats_gate",
      "params": {
        "event": "bullish_engulfing",
        "target": "next_bullish",
        "metric": "p_hat",
        "comparator": ">=",
        "threshold": 0.55,
        "condition_name": "session",
        "condition_value": "london",
        "min_samples": 500,
        "symbol": "EURUSD",
        "timeframe": "1m",
        "allow_if_missing": false,
        "allow_if_insufficient": false
      }
    }
  ]
}
```

### Example: stats_gate_score for weighting (no hard block)

```json
{
  "filters": [
    {
      "type": "stats_gate",
      "params": {
        "event": "bullish_engulfing",
        "target": "next_bullish",
        "metric": "p_hat",
        "comparator": ">=",
        "threshold": 0.50,
        "symbol": "EURUSD",
        "timeframe": "1m",
        "allow_if_missing": true,
        "allow_if_insufficient": true
      }
    }
  ]
}
```

Note: `stats_gate_score` is a helper (not a filter type in JSON). Use it in code
when you want a [0-1] score for weighting instead of a hard gate.
