# Overview du module Levels

Le module Levels calcule et persiste des niveaux de structure de marche lents
dans `marketdata.levels`, afin que les backtests Python, le stack Java et le
frontend Angular reutilisent les memes signaux.

## Levels supportes

Les detecteurs actuels couvrent :

- **PDH/PDL** : previous day high/low, ancre sur la bougie journaliere UTC closee.
- **PWH/PWL** : previous ISO week high/low.
- **PMH/PML** : previous month high/low.
- **GAP_D / GAP_W** : zones de gap entre la close et la prochaine open en daily/weekly.
- **FVG** : fair value gaps a 3 bougies (bullish/bearish) sur la timeframe de base.
- **FVG_HTF** : FVG higher-timeframe (H1/H4/D1) via resampling, ancre sur la
  bougie centrale.
- **POC** : point-of-control simplifie via histogramme.
- **SWING_H / SWING_L** : pivots fractals n-bar capturant les swings.
- **EQH / EQL** : equal highs/lows (liquidity pools) via bandes de tolerance.
- **BOS_H / BOS_L / MSS** : Break of Structure et Market Structure Shift, declenches
  quand la close casse le dernier swing.
- **VWAP_SESSION / VWAP_DAY / VWAP_WEEK** : VWAP ancrees (developing/fixed) avec
  bandes sigma optionnelles `VWAP_BAND_{k}+`.
- **ADR_BAND_k** : enveloppes d'Average Daily Range autour de l'open de session.
- **PIVOT_P / PIVOT_R1..R3 / PIVOT_S1..S3** : niveaux pivot classiques (floor pivots)
  derives de la session precedente.

Les round numbers (RN) peuvent aussi etre generes statiquement.

## Ajouts Phase 1.5

- **Fills** : `valid_to_ts` est renseigne sur la premiere close qui touche la
  zone GAP ou FVG (logique MVP). Les iterations suivantes ajouteront un mode
  d'overlap intrabar complet.
- **Endpoints** :
  - `POST /levels/fill` rafraichit les fills FVG/GAP. Le body est un
    `LevelsBuildSpec` qui fournit la source et le range.
  - `GET /levels/active` retourne les zones ouvertes (`valid_to_ts IS NULL`) filtrees
    par symbol, types et fenetre de date optionnelle.
- **Nouveaux levels** : session highs/lows, opening range (ORH/ORL), initial
  balance (IBH/IBL) et previous open/close pour daily/weekly/monthly
  (PDO/PDC, PWO/PWC, PMO/PMC).
- **Ajouts Phase 2B** : VWAP ancrees (session/day/week) avec bandes sigma,
  FVG higher-timeframe via resampling, ADR envelopes et floor pivots journaliers.
- **Configuration** : les session windows et les durees Opening Range/Initial
  Balance sont configurables via `session_windows` et `orib` dans `LevelsBuildSpec`.

## Structure (Phase 2A)

La Phase 2A introduit des primitives de structure basees sur les fractal swings
et les liquidity pools :

- `SWING_H` / `SWING_L` utilisent des fractals n-bar (config via `left`/`right`)
  pour ancrer les swings sur la timeframe de base.
- `EQH` / `EQL` clusterisent les highs/lows proches sur une fenetre configurable
  et retournent des zones `[price_lo, price_hi]` ancrees sur le dernier touch.
- `BOS_H` / `BOS_L` se declenchent quand la close casse le swing le plus recent
  dans la direction, et `MSS` marque un shift quand la cassure inverse le run.

Les helpers dans `quant_engine.levels.helpers` facilitent la consommation en stats
et backtests :

```python
from quant_engine.levels import helpers as lvl_helpers, repo

levels = repo.select_levels(engine, "marketdata.levels", symbol="EURUSD", level_types=["EQH"], active_only=False)
df = lvl_helpers.join_levels(ohlcv_df, levels)
in_pool = lvl_helpers.in_zone(ohlcv_df, levels, "EQH", tolerance=0.0001)
distance = lvl_helpers.distance_to(ohlcv_df, levels, "EQH", side="edge")
recent_touch = lvl_helpers.touched_since(ohlcv_df, levels, "EQH", bars=5)
```

Les conditions stats encapsulent ces helpers via `in_zone_level`, `distance_to_level`
et `touched_level_since`, en chargeant automatiquement les levels persistes depuis
`marketdata.levels`.

## Idempotence & perf

- Chaque ligne porte un `uniq_hash` deterministe (SHA-256) base sur `symbol`,
  `level_type`, `timeframe`, prix arrondis, `anchor_ts`, `valid_from_ts` optionnel
  et `params_hash`. L'index unique assure l'idempotence.
- Index b-tree additionnels sur `(symbol, level_type, anchor_ts)` et
  `(symbol, valid_from_ts, valid_to_ts)` pour accelerer les scans, overlays et
  checks de validite.
- Deux views exposent uniquement les lignes actives (zones encore ouvertes) pour
  les consumers sensibles a la latence (Java execution stack, Angular UI).

Exemple de requete pour les zones ouvertes :

```sql
SELECT *
FROM marketdata.view_levels_active_zones
WHERE symbol = 'EURUSD'
  AND level_type IN ('FVG', 'GAP_D');
```

## Execution des detections

### CLI

```bash
export QE_MARKETDATA_MYSQL_URL='mysql+pymysql://py_user:***@mysql:3306/marketdata'
poetry run qe levels build --spec specs/levels_example.json
```

### API

Demarrer l'API localement :

```bash
poetry run uvicorn quant_engine.api.app:app --reload --port 8000
```

Declencher un build et recuperer les levels :

```bash
curl -X POST "http://localhost:8000/levels/build" \
  -H "Content-Type: application/json" \
  -d @specs/levels_example.json

curl "http://localhost:8000/levels?symbol=EURUSD&level_type=PDH&limit=50"

curl "http://localhost:8000/levels/search?symbol=EURUSD&type=EQH,EQL&limit=20"
```

L'endpoint `/levels/nearest` retourne les levels les plus proches d'un prix cible,
utile pour les overlays live :

```bash
curl "http://localhost:8000/levels/nearest?symbol=EURUSD&price=1.0825&limit=10"
```

## Notes

- Tous les timestamps sont normalises en UTC et references sur la close de la
  periode agregee (daily, weekly, monthly).
- Les GAP capturent la range entre la close precedente et l'open suivante.
- Le detecteur FVG MVP ne gere pas encore l'invalidation (TODOs en place).
- Le POC utilise un fallback histogramme adapte au FX spot (volume peu fiable).
  Un vrai volume profile sera ajoute dans une iteration future.
