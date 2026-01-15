# Filter reference

Ce document fournit une reference concise des pre-trade filters disponibles dans le Quant Engine.

## Backtest and DCA availability

Le tableau ci-dessous resume quels filters peuvent etre utilises en backtest et en DCA lorsqu'ils sont actives dans les specs JSON.
Si des donnees requises manquent, certains filtres deviennent des no-op (retournent True partout), tandis que
d'autres loguent une erreur et stoppent l'execution.

| Filter | Backtest | DCA | One-line summary |
| --- | --- | --- | --- |
| adx | Yes | Yes | Trend strength above threshold (ADX). |
| atr | Yes | Yes | ATR/close within a min/max band. |
| ema_slope | Yes | Yes | EMA slope above threshold. |
| volume_surge | Yes (requires volume) | Yes (requires volume) | Detect volume spikes (z-score or ratio). |
| vwap_side | Yes (levels optional) | Yes (levels optional) | Price above/below anchored VWAP. |
| poc_distance | Yes (levels required) | Yes (levels required) | Price within distance of active POC. |
| liquidity_sweep | Yes | Yes | Sweep recent highs/lows (ICT). |
| bos | Yes (levels optional) | Yes (levels optional) | Break of structure up/down. |
| mss | Yes (levels optional) | Yes (levels optional) | Market structure shift (BOS flip). |
| session_time | Yes | Yes | Allow trades in a session window. |
| day_of_week | Yes | Yes | Allow/deny specific weekdays. |
| day_of_month | Yes | Yes | Allow first/last N days of month. |
| month_of_year | Yes | Yes | Allow/deny specific months. |
| intraday_time | Yes | Yes | Allow trades in time-of-day window. |
| k_consecutive | Yes | Yes | K consecutive candles in same direction. |
| seasonality_bin | Yes (stats optional) | Yes (stats optional) | Allow specific seasonal bins. |
| hurst_regime | Yes | Yes | Hurst exponent regime filter. |
| entropy_window | Yes | Yes | Directional entropy window filter. |
| daily_loss_cap | Conditional (needs pnl/equity/ret cols) | Conditional (needs pnl/equity/ret cols) | Lockout after daily loss cap. |
| daily_trades_cap | Conditional (needs signal col) | Conditional (needs signal col) | Limit number of entries per day. |
| cooldown_bars | Conditional (needs signal col) | Conditional (needs signal col) | Cooldown after each signal. |
| atr_risk_gate | Yes | Yes | Block when ATR/close too high. |
| equity_dd_lockout | Conditional (needs equity col) | Conditional (needs equity col) | Lockout after equity drawdown. |
| benford_law | Yes | Yes | Benford MSE below threshold. |
| cycles | Yes | Yes | Low dominant autocorrelation (cycles). |
| donchian_channels | Yes | Yes | Donchian breakout filter. |
| liquidity_cmf | Yes (requires volume) | Yes (requires volume) | Chaikin Money Flow threshold. |
| statistical_arbitrage | Yes | Yes | Omega/Info ratio threshold. |
| psychologic_ulcer | Yes | Yes | Ulcer index below threshold. |
| stationarity | Yes | Yes | Low lag-1 autocorrelation. |
| volatility | Yes | Yes | Entropy (and optional ATR) threshold. |
| ema_structure | Yes | Yes | EMA stack and trend validation. |
| rsi_entry | Yes | Yes | RSI threshold filter. |
| macd_entry | Yes | Yes | MACD cross filter. |
| volume_above_average | Yes (requires volume) | Yes (requires volume) | Volume above its rolling average. |
| fractal_analysis | Yes | Yes | Hurst + skew/kurtosis bounds. |
| mean_reversion | Yes | Yes | BB + Keltner mean-reversion trigger. |
| contradictory_signals | Yes | Yes | Block conflicting indicator signals. |
| linear_regression_macd_cross | Yes | Yes | Predict MACD cross via regression. |
| atr_rising | Yes | Yes | ATR rising vs previous bar/lookback. |
| market_regime | Yes | Yes | Detect trend/range/compression regimes. |
| trend | Yes | Yes | Trend direction via HH/HL or EMA. |
| biais_institutional | Yes | Yes | EMA/VWAP + optional macro filters. |
| stats_gate | Yes | Yes | Gate using persisted market stats. |
| mtf_anomaly | Yes | Yes | Block when anomalies appear on both timeframes. |
| market_manipulation | Yes | Yes | Block when entropy/kurtosis/volatility indicate anomalies. |
| htf_poi | Yes | Yes | Price interacts with active HTF zones/POI levels. |
| orderflow_delta | Yes (requires buy/sell or delta) | Yes (requires buy/sell or delta) | Orderflow delta/ratio filter. |
| macro_cot_oi | Yes (optional columns) | Yes (optional columns) | Macro COT/OI alignment filter. |
| lower_timeframe_confluence | Yes | Yes | Confluence score across momentum/ADX/VWAP/delta/EMA. |
| psychologic_and_news | Yes | Yes | News/psychologic blackout windows. |

## Trend & Volatility filters

Ces filtres exploitent des indicateurs de tendance ou de volatilité calculés directement à partir du flux OHLCV.

### `adx`
- **Paramètres** : `window` (int), `thresh` (float).
- **Retour** : `True` si `ADX(window) > thresh`.
- **Utilité** : confirme qu'une tendance est suffisamment forte pour éviter les phases de range.

### `atr`
- **Parametres** : `window` (int), `min_mult` (float), `max_mult` (float) en pourcentage.
- **Retour** : `True` si `ATR(window) / close` appartient a l'intervalle `[min_mult, max_mult]` (valeurs en %).
- **Utilite** : filtre les marches trop calmes ou, a l'inverse, trop explosifs.

### `ema_slope`
- **Paramètres** : `window` (int), `slope_thresh` (float).
- **Retour** : `True` si la pente de `EMA(window)` est `> slope_thresh` (tendance haussière) ou `< -slope_thresh` (tendance baissière).
- **Utilité** : valide que la tendance présente une pente marquée plutôt qu'une moyenne mobile plate.

## Volume & Market Profile filters

### `volume_surge`
- **Paramètres** : `window` (int), `mode` = `"z"` \| `"ratio"`, `z_thresh`, `ratio_thresh`.
- **Retour** : `True` si pic de volume (z-score≥seuil ou ratio≥seuil).
- **Notes** : si `volume` absent → renvoie `False` partout.

### `vwap_side`
- **Paramètres** : `anchor` = `"day"` \| `"session"`, `side` = `"above"` \| `"below"`, `from_levels` (bool), `symbol`.
- **Retour** : `True` si `close` est du bon côté du VWAP ancré.
- **Sources** : tente de lire `VWAP_DAY` / `VWAP_SESSION` depuis `marketdata.levels`, sinon **fallback** VWAP journalier local.
- **Avertissement** : la qualité dépend du volume dispo ; en FX spot, le VWAP peut être approximatif.

### `poc_distance`
- **Paramètres** : `max_distance` (float), `unit` = `"abs"` \| `"pct"`, `symbol`, `level_type="POC"`.
- **Retour** : `True` si la distance au **POC actif** le plus proche ≤ seuil.
- **Notes** : nécessite des **POC** en DB (`marketdata.levels`). Sans POC: renvoie `False`.

## Structure & ICT filters

### `liquidity_sweep`
- **Idée** : mèche qui “prend” la liquidité d’un extrême récent (EQH/EQL local).  
- **Paramètres** :  
  - `side` = `"high"` \| `"low"`  
  - `lookback` (barres)  
  - `require_close_back_in` (bool, par défaut True)  
  - `tolerance_ticks`, `price_increment` (pour marge)  
- **Retour** : `True` quand la barre réalise un sweep (prise + close-back-in si demandé).  
- **Source** : pure OHLC (pas besoin de DB).

### `bos`
- **Idée** : cassure du dernier swing high/low → Break Of Structure.  
- **Paramètres** :  
  - `direction` = `"up"` \| `"down"`  
  - `left`, `right` (fractals)  
  - `use_levels` (True par défaut) + `symbol` → si SWING_H/L présents en DB, les utiliser ; sinon fallback fractals.  
- **Retour** : `True` sur la barre qui casse.  

### `mss`
- **Idée** : “flip” de structure : cassure dans un sens, puis cassure opposée dans `window` barres.
- **Paramètres** : `left`, `right`, `window`, `use_levels`, `symbol`.
- **Retour** : `True` sur la barre qui réalise la deuxième cassure.

## Seasonality & Time filters

### `session_time`
- **Sessions pré-définies (locales, DST-aware)** :
  - Asia (Tokyo) : 09:00–17:00 JST
  - London : 08:00–16:30 Europe/London
  - New York : 09:30–16:00 America/New_York
- **Paramètres** : `session` (`"asia"`, `"london"`, `"newyork"`), `tz` (optionnel pour forcer la timezone).
- **Retour** : `True` si la barre tombe dans la session choisie.

### `day_of_week`
- **Paramètres** : `allowed_days` (liste d'entiers 0–6) ou `blocked_days`.
- **Retour** : `True` si le jour de semaine est autorisé.
- **Exemple** : exclure lundi (0) et vendredi (4) → `blocked_days=[0,4]`.

### `day_of_month`
- **Paramètres** : `mode="first"|"last"`, `n` (entier > 0).
- **Retour** : `True` si la barre se situe dans les `n` premiers ou derniers jours du mois.

### `month_of_year`
- **Paramètres** : `allowed_months` ou `blocked_months` (1–12).
- **Retour** : `True` si le mois est autorisé (ou non bloqué).
- **Exemple** : éviter août (8) et décembre (12) → `blocked_months=[8,12]`.

### `intraday_time`
- **Paramètres** : `start`, `end` (format `"HH:MM"`), `tz`.
- **Retour** : `True` si la barre tombe dans la fenêtre `[start, end)`.
- **Exemple** : ne garder que 09:00–12:00 UTC.

## Statistical & Probabilistic filters

### `k_consecutive`
- **Idée** : éviter d’entrer après trop de bougies alignées, ou au contraire détecter une séquence momentum.
- **Paramètres** : `k`, `direction` (`"up"`/`"down"`), `use_body`.
- **Retour** : `True` si la fenêtre courante présente `k` bougies successives dans la direction souhaitée.

### `seasonality_bin`
- **Idée** : n’activer que certains bins temporels.
- **Modes** : `hour` (0–23), `dow` (0–6), `dom` (1–31), `month` (1–12), `session` (0–3).
- **Paramètres** : `allowed` / `blocked` ou `min_winrate` (+ `symbol`) pour utiliser `quant.seasonality_profiles`.
- **Retour** : `True` si la barre est dans un bin “positif”.

### `hurst_regime`
- **Idée** : filtrer par régime (trend/mean-revert/noise).
- **Paramètres** : `window`, `min_h`, `max_h`.
- **Retour** : `True` si `H` appartient à l’intervalle `[min_h, max_h]`.

### `entropy_window`
- **Idée** : mesurer la “randomness” directionnelle locale.
- **Paramètres** : `window`, `max_entropy` et/ou `min_entropy`, `use_body`.
- **Retour** : `True` si l’entropie respecte le(s) seuil(s).

## Risk & Money Management filters

Ces filtres visent à plafonner les pertes, limiter le nombre d’entrées et adapter le risque aux conditions de marché.

### `daily_loss_cap`
- **Idée** : couper les signaux quand la perte journalière dépasse un plafond.
- **Paramètres** : `loss_cap` (float, en devise), `mode` = `"pnl"` \| `"equity"` \| `"ret_notional"`, `pnl_col` / `equity_col` / `ret_col` (+ `notional`).
- **Comportement** : renvoie `False` (lock) pour le reste de la journée dès que la perte cumulée ≤ `-loss_cap`.
- **Tolérance** : si les colonnes requises sont absentes, le filtre renvoie `True` partout (no-op).

### `daily_trades_cap`
- **Idée** : limiter le nombre d’entrées par jour.
- **Paramètres** : `signal_col` (bool), `max_trades_per_day` (int).
- **Notes** : nécessite que `signal_col` existe déjà (par exemple le résultat d’un bloc de règles). Après avoir atteint la limite, le filtre reste `False` jusqu’à la fin de la journée.

### `cooldown_bars`
- **Idée** : imposer un cooldown après un signal.
- **Paramètres** : `signal_col` (bool), `cooldown_bars` (int).
- **Retour** : `True` uniquement si au moins `cooldown_bars` barres se sont écoulées depuis le dernier `True`.

### `atr_risk_gate`
- **Idée** : bloquer si la volatilité relative (ATR/close) est trop élevée.
- **Paramètres** : `atr_window`, `max_atr_pct`.
- **Utilité** : conserver un sizing cohérent et un ratio rendement/risque praticable. Si les colonnes OHLC sont absentes, le filtre devient un no-op (`True`).

### `equity_dd_lockout`
- **Idée** : lockout si le drawdown de l’equity dépasse un seuil.
- **Paramètres** : `equity_col`, `max_dd_pct`.
- **Notes** : nécessite une colonne equity ; sinon le filtre renvoie `True` (no-op).

## Additional filters

### `benford_law`
- **Parametres** : `window`, `series_type`, `metric`, `mad_threshold`, `chi2_threshold`, `atr_window`,
  `price_col`, `open_col`, `high_col`, `low_col`, `volume_col`.
- **Series_type** : `range`, `body`, `volume`, `delta_range`, `returns`, `atr`, `wick`.
- **Metric** : `mad`, `chi2`, `both`.
- **Retour** : `True` si l'anomalie Benford reste sous les seuils.

### `cycles`
- **Parametres** : `window`, `max_lag`, `max_r2`, `price_col`.
- **Retour** : `True` si la force du cycle dominant reste faible.

### `donchian_channels`
- **Parametres** : `window`, `direction`, `high_col`, `low_col`, `close_col`.
- **Retour** : `True` sur breakout Donchian dans la direction choisie.

### `liquidity_cmf`
- **Parametres** : `window`, `threshold`, `high_col`, `low_col`, `close_col`, `volume_col`.
- **Retour** : `True` si le CMF depasse le seuil.

### `statistical_arbitrage`
- **Parametres** : `window`, `omega_thresh`, `info_thresh`, `price_col`, `require_info`.
- **Retour** : `True` si omega (et optionnellement info ratio) depassent les seuils.

### `psychologic_ulcer`
- **Parametres** : `window`, `max_ulcer`, `price_col`.
- **Retour** : `True` si l'Ulcer Index est sous le seuil.

### `stationarity`
- **Parametres** : `window`, `max_abs_autocorr`, `price_col`, `method`, `adf_pvalue`, `kpss_pvalue`, `allow_if_missing`.
- **Method** : `acf` (default), `adf`, `kpss`, `both`.
- **Retour** : `True` si le test de stationnarite passe.


### `volatility`
- **Parametres** : `window`, `max_entropy`, `atr_window`, `max_atr_pct`, `bb_window`, `bb_max_width`,
  `hv_window`, `max_hv`, `vix_window`, `max_vix`.
- **Retour** : `True` si l'entropie (et les extras de volatilite) restent sous les seuils.
- **Notes** : VIX approx et HV sont bases sur les rendements log. Pas de volatilite implicite.


### `market_manipulation`
- **Parametres** : `window`, `entropy_threshold`, `kurtosis_threshold`, `atr_window`, `atr_mult`, `require_all`,
  `high_col`, `low_col`, `close_col`.
- **Retour** : `True` si les signaux d'entropie/kurtosis/volatilite explosive ne sont pas simultanement (ou cumulativement) declenches.
- **Notes** : utilise l'entropie directionnelle, la kurtosis des retours, et un spike de true range vs ATR.


### `htf_poi`
- **Parametres** : `level_types`, `symbol`, `mode` ("in_zone"|"distance"), `tolerance`,
  `max_distance`, `distance_unit`, `require_all`, `allow_if_missing`, `allow_if_empty`, `close_col`.
- **Retour** : `True` si le prix interagit avec au moins une zone HTF active.
- **Notes** : s'appuie sur `marketdata.levels` (ex: FVG_HTF, GAP_D, PDH/PDL). Si la DB est indisponible et
  `allow_if_missing=true`, le filtre devient un no-op.


### `orderflow_delta`
- **Parametres** : `delta_col`, `buy_col`, `sell_col`, `volume_col`, `mode` ("delta"|"ratio"),
  `window`, `min_delta`, `min_ratio`, `zscore_thresh`, `side`, `allow_if_missing`, `require_positive`.
- **Retour** : `True` si le delta (ou ratio delta/volume) respecte les seuils.
- **Notes** : si les colonnes requises sont absentes et `allow_if_missing=true`, le filtre devient un no-op.


### `macro_cot_oi`
- **Parametres** : `cot_col`, `oi_col`, `cot_bias_threshold`, `oi_change_threshold`,
  `side`, `require_all`, `allow_if_missing`.
- **Retour** : `True` si les signaux COT/OI sont alignes avec la direction demandee.
- **Notes** : si les colonnes requises sont absentes et `allow_if_missing=true`, le filtre devient un no-op.


### `lower_timeframe_confluence`
- **Parametres** : `side`, `momentum_period`, `min_momentum`, `adx_window`, `adx_thresh`,
  `vwap_max_dev`, `vwap_side`, `vwap_price_col`, `vwap_volume_col`,
  `ema_fast`, `ema_slow`, `ema_long`,
  `delta_col`, `buy_col`, `sell_col`, `volume_col`, `delta_min`, `delta_ratio`,
  `min_score`, `min_score_pct`, `require_all`, `allow_if_missing`,
  `close_col`, `high_col`, `low_col`.
- **Retour** : `True` si le score de confluence atteint le seuil.
- **Notes** : les composantes actives sont agregees (score) ou require_all.


### `psychologic_and_news`
- **Parametres** : `news_times`, `news_col`, `pre_minutes`, `post_minutes`, `allow_if_missing`.
- **Retour** : `True` si la barre est hors fenetre d'impact news.
- **Notes** : `news_col` peut etre une colonne bool (True=impact). Sinon `news_times` (ISO) est utilise.
  Les signaux de sentiment/comportement ne sont pas encore integres.
### `ema_structure`
- **Parametres** : `ema_fast`, `ema_slow`, `ema_long`, `require_close_above_slow`, `require_fast_rising`.
- **Retour** : `True` si la structure EMA est haussiere.

### `rsi_entry`
- **Parametres** : `window`, `threshold`, `direction`.
- **Retour** : `True` si RSI depasse (ou passe sous) le seuil.

### `macd_entry`
- **Parametres** : `fast`, `slow`, `signal`, `direction`.
- **Retour** : `True` sur cross MACD directionnel.

### `volume_above_average`
- **Parametres** : `window`, `multiplier`, `volume_col`.
- **Retour** : `True` si le volume est au-dessus de sa moyenne.

### `fractal_analysis`
- **Parametres** : `window`, `min_h`, `max_h`, `max_abs_skew`, `max_kurtosis`.
- **Retour** : `True` si Hurst (et optionnellement skew/kurtosis) respectent les bornes.

### `mean_reversion`
- **Parametres** : `window`, `bb_mult`, `keltner_mult`, `atr_window`, `direction`.
- **Retour** : `True` si le prix est hors BB + Keltner.

### `contradictory_signals`
- **Parametres** : `rsi_window`, `stoch_window`, `williams_window`, `z_window`, `ema_fast`, `ema_slow`, `max_conflicts`.
- **Retour** : `True` si les contradictions indicateurs restent sous le seuil.

### `linear_regression_macd_cross`
- **Parametres** : `fast`, `slow`, `signal`, `lookback`, `direction`.
- **Retour** : `True` si un cross MACD est predit par regression.

### `atr_rising`
- **Parametres** : `window`, `lookback`.
- **Retour** : `True` si l'ATR est en hausse.

### `market_regime`
- **Parametres** : `regime`, `adx_window`, `adx_thresh`, `atr_window`, `max_atr_pct`, `bb_window`, `bb_width_thresh`.
- **Retour** : `True` si le regime detecte correspond.

### `trend`
- **Parametres** : `direction`, `lookback`, `method`, `ema_fast`, `ema_slow`.
- **Retour** : `True` si la tendance correspond.

### `biais_institutional`
- **Parametres** : `ema_fast`, `ema_slow`, `price_col`, `volume_col`, `vwap_side`,
  `cot_col`, `oi_col`, `cot_bias_threshold`, `oi_min_change`.
- **Retour** : `True` si les EMA/VWAP sont alignes, avec macro optionnelle (COT/OI).

### `stats_gate`
- **Parametres** : `event`, `target`, `threshold`, `metric`, `comparator`, `split`,
  `condition_name`, `condition_params`, `condition_value`, `min_samples`,
  `symbol`, `timeframe`, `allow_if_missing`, `allow_if_insufficient`.
- **Retour** : `True` si la stat en base respecte le seuil (fallback global automatique).

### `stats_gate_score`
- **Parametres** : `event`, `target`, `metric`, `split`, `condition_name`, `condition_params`,
  `condition_value`, `min_samples`, `symbol`, `timeframe`, `allow_if_missing`,
  `allow_if_insufficient`, `scale_min`, `scale_max`.
- **Retour** : Score [0-1] normalise pour pondération (utilisation hors filtre booléen).
  - **Note** : helper uniquement (pas expose dans `filters_registry`, donc pas utilisable dans une spec JSON).


### `mtf_anomaly`
- **Parametres** : `higher_timeframe`, `base_window`, `higher_window`, `metric="benford_mad"|"entropy"`,
  `series_type`, `mad_threshold`, `entropy_threshold`, `atr_window`, `require_both`.
- **Retour** : `True` sauf si l'anomalie est detectee sur la timeframe courante ET la timeframe superieure.

## Pending filters (not implemented)

Les filtres ci-dessous restent a implementer si tu veux la parite complete:
- ICTPointOfInterestFilter (order blocks, continuation/breakaway gaps, fib retracements, psychological levels, news open gaps)
- OrderBookLiquidityFilter (absorption, spoofing, stacking)
- TradeFilterService (weighted scoring orchestration)
- FilterRuleAdapter (rule glue / tolerance)
- DynamicStopLossRule (exit logic)

