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
