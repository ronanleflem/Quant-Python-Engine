# Filter reference

This document provides a concise reference for the pre-trade filters available in the Quant Engine.

## Trend & Volatility filters

Ces filtres exploitent des indicateurs de tendance ou de volatilité calculés directement à partir du flux OHLCV.

### `adx`
- **Paramètres** : `window` (int), `thresh` (float).
- **Retour** : `True` si `ADX(window) > thresh`.
- **Utilité** : confirme qu'une tendance est suffisamment forte pour éviter les phases de range.

### `atr`
- **Paramètres** : `window` (int), `min_mult` (float), `max_mult` (float).
- **Retour** : `True` si `ATR(window) / close` appartient à l'intervalle `[min_mult, max_mult]`.
- **Utilité** : filtre les marchés trop calmes ou, à l'inverse, trop explosifs.

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
- **Sessions pré-définies (UTC)** :
  - Asia : 23:00–07:00
  - London : 07:00–15:00
  - NewYork : 13:00–21:00
- **Paramètres** : `session` (`"asia"`, `"london"`, `"newyork"`), `tz` (timezone de référence).
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
