# Migration des wrappers legacy `quant_engine.stats.*`

Ce document décrit la migration progressive des wrappers legacy vers les modules Market Intelligence canoniques, sans rupture immédiate.

## Statut

- Les wrappers `quant_engine.stats.conditions.*` et `quant_engine.stats.events.*` restent utilisables pour la compatibilité.
- Chaque appel déclenche désormais un `DeprecationWarning` explicite.
- Cible de suppression : **v0.15.0** (date cible : **2026-04-30**).

## Migration old -> new

### Conditions

- `quant_engine.stats.conditions.htf_trend` -> `quant_engine.market_intelligence.conditions.htf_trend`
- `quant_engine.stats.conditions.vol_tertile` -> `quant_engine.market_intelligence.conditions.vol_tertile`
- `quant_engine.stats.conditions.session` -> `quant_engine.market_intelligence.conditions.session`
- `quant_engine.stats.conditions.hour_bin` -> `quant_engine.market_intelligence.conditions.hour_bin`
- `quant_engine.stats.conditions.day_of_week` -> `quant_engine.market_intelligence.conditions.day_of_week`
- `quant_engine.stats.conditions.month_of_year` -> `quant_engine.market_intelligence.conditions.month_of_year`
- `quant_engine.stats.conditions.session_from_ts` -> `quant_engine.market_intelligence.conditions.session_from_ts`

### Events

- `quant_engine.stats.events.k_consecutive` -> `quant_engine.market_intelligence.events.k_consecutive`
- `quant_engine.stats.events.shock_atr` -> `quant_engine.market_intelligence.events.shock_atr`
- `quant_engine.stats.events.breakout_hhll` -> `quant_engine.market_intelligence.events.breakout_hhll`
- `quant_engine.stats.events.bullish_candle` -> `quant_engine.market_intelligence.events.bullish_candle`
- `quant_engine.stats.events.bearish_candle` -> `quant_engine.market_intelligence.events.bearish_candle`
- `quant_engine.stats.events.bullish_engulfing` -> `quant_engine.market_intelligence.events.bullish_engulfing`
- `quant_engine.stats.events.bearish_engulfing` -> `quant_engine.market_intelligence.events.bearish_engulfing`
- `quant_engine.stats.events.bullish_streak` -> `quant_engine.market_intelligence.events.bullish_streak`
- `quant_engine.stats.events.bearish_streak` -> `quant_engine.market_intelligence.events.bearish_streak`
- `quant_engine.stats.events.gap_up` -> `quant_engine.market_intelligence.events.gap_up`
- `quant_engine.stats.events.gap_down` -> `quant_engine.market_intelligence.events.gap_down`
- `quant_engine.stats.events.always_true` -> `quant_engine.market_intelligence.events.always_true`

## Exemple de warning

Message attendu :

`quant_engine.stats.events.gap_up is deprecated and will be removed in v0.15.0 (target date: 2026-04-30); use quant_engine.market_intelligence.events instead.`

## Recommandation projet

- Remplacer les imports legacy dans les nouvelles features.
- Conserver temporairement les wrappers uniquement pour compatibilité ascendante.
