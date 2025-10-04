# Filter reference

This document provides a concise reference for the pre-trade filters available in the Quant Engine.

## Core volatility & trend filters

These filters rely on volatility or trend metrics computed directly from the OHLCV stream. Refer to the individual function docstrings for the latest parameter details.

- `adx`
- `atr`
- `ema_slope`

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
