# Levels module overview

The levels module computes and persists slow-moving market structure levels in
`marketdata.levels` so that Python backtests, the Java execution stack and the
Angular frontend can reuse the same signals.

## Supported levels

The current detectors cover the following level types:

* **PDH/PDL** – previous day high/low anchored on the completed UTC daily bar.
* **PWH/PWL** – previous ISO week high/low.
* **PMH/PML** – previous month high/low.
* **GAP_D / GAP_W** – gap zones between close and next open on the daily/weekly
  aggregate.
* **FVG** – three-candle fair value gaps (bullish and bearish) on the base
  timeframe.
* **FVG_HTF** – higher-timeframe fair value gaps built from resampled H1/H4/D1
  candles while keeping the anchor on the centre bar.
* **POC** – simplified point-of-control using histogram counts.
* **SWING_H / SWING_L** – n-bar fractal pivots capturing market structure
  swings.
* **EQH / EQL** – equal highs/lows (liquidity pools) detected via tolerance
  bands on recent extremes.
* **BOS_H / BOS_L / MSS** – Break of Structure and Market Structure Shift events
  triggered when closes breach the latest swing.
* **VWAP_SESSION / VWAP_DAY / VWAP_WEEK** – anchored VWAP curves (developing
  and fixed) with optional sigma bands `VWAP_BAND_{k}+`.
* **ADR_BAND_k** – daily Average Daily Range envelopes around the current
  session open.
* **PIVOT_P / PIVOT_R1..R3 / PIVOT_S1..S3** – classic floor pivot levels derived
  from the previous session’s range.

Round numbers (RN) can also be generated statically for convenience.

## Phase 1.5 additions

* **Fills:** `valid_to_ts` is populated on the first closing price that touches
  the gap or fair value gap zone (MVP logic). Future iterations will add a full
  intrabar overlap mode.
* **Endpoints:**
  * `POST /levels/fill` refreshes FVG/GAP fills. The body is a
    `LevelsBuildSpec` providing the data source and date range.
  * `GET /levels/active` returns open zones (`valid_to_ts IS NULL`) filtered by
    symbol, level types and optional date window.
* **New levels:** session highs/lows, opening range (ORH/ORL), initial balance
  (IBH/IBL) and previous open/close levels for daily/weekly/monthly periods
  (PDO/PDC, PWO/PWC, PMO/PMC).
* **Phase 2B additions:** anchored VWAP (session/day/week) with sigma bands,
  higher-timeframe FVGs via resampling, ADR envelopes and daily floor pivots.
* **Configuration:** session windows and Opening Range/Initial Balance durations
  are configurable via the `session_windows` and `orib` sections of the
  `LevelsBuildSpec`.

## Structure (Phase 2A)

Phase 2A introduces higher-level structure primitives built on fractal swings
and liquidity pools:

* `SWING_H` / `SWING_L` use n-bar fractals (configurable via `left`/`right`) to
  anchor swings on the base timeframe.
* `EQH` / `EQL` cluster nearly equal highs/lows inside a configurable lookback
  window, returning narrow zones `[price_lo, price_hi]` anchored on the latest
  touch.
* `BOS_H` / `BOS_L` fire when the closing price breaks the most recent swing in
  the corresponding direction, while `MSS` marks a shift when the break
  reverses the previous run.

The new helpers in `quant_engine.levels.helpers` simplify consumption inside
stats and backtests:

```python
from quant_engine.levels import helpers as lvl_helpers, repo

levels = repo.select_levels(engine, "marketdata.levels", symbol="EURUSD", level_types=["EQH"], active_only=False)
df = lvl_helpers.join_levels(ohlcv_df, levels)
in_pool = lvl_helpers.in_zone(ohlcv_df, levels, "EQH", tolerance=0.0001)
distance = lvl_helpers.distance_to(ohlcv_df, levels, "EQH", side="edge")
recent_touch = lvl_helpers.touched_since(ohlcv_df, levels, "EQH", bars=5)
```

Statistics conditions wrap these helpers via `in_zone_level`, `distance_to_level`
and `touched_level_since`, automatically loading persisted levels from
`marketdata.levels`.

## Idempotence & perf

* Every row carries a deterministic `uniq_hash` (SHA-256) built from
  `symbol`, `level_type`, `timeframe`, rounded prices, anchor timestamp,
  optional `valid_from_ts` and the detector `params_hash`. The hash is
  enforced via a unique index so repeated ingestions stay idempotent.
* Additional b-tree indices on `(symbol, level_type, anchor_ts)` and
  `(symbol, valid_from_ts, valid_to_ts)` keep the most common lookups quick for
  scans, overlays and validity checks.
* Two helper views expose only active rows (still-open points or zones) for
  latency-sensitive consumers like the Java execution stack or the Angular UI.

Example query for open zones:

```sql
SELECT *
FROM marketdata.view_levels_active_zones
WHERE symbol = 'EURUSD'
  AND level_type IN ('FVG', 'GAP_D');
```

## Running detections

### CLI

```bash
export QE_MARKETDATA_MYSQL_URL='mysql+pymysql://py_user:***@mysql:3306/marketdata'
poetry run qe levels build --spec specs/levels_example.json
```

### API

Start the API locally:

```bash
poetry run uvicorn quant_engine.api.app:app --reload --port 8000
```

Trigger a build and fetch persisted levels:

```bash
curl -X POST "http://localhost:8000/levels/build" \
  -H "Content-Type: application/json" \
  -d @specs/levels_example.json

curl "http://localhost:8000/levels?symbol=EURUSD&level_type=PDH&limit=50"

curl "http://localhost:8000/levels/search?symbol=EURUSD&type=EQH,EQL&limit=20"
```

The `/levels/nearest` endpoint returns levels closest to a target price, making
it convenient for live overlays:

```bash
curl "http://localhost:8000/levels/nearest?symbol=EURUSD&price=1.0825&limit=10"
```

## Notes

* All timestamps are normalised to UTC and reference the close of the
  aggregated period (daily, weekly, monthly).
* Gap zones capture the range between the previous close and the next open.
* The MVP FVG detector does not yet handle invalidation; TODOs mark the areas
  earmarked for refinement.
* The POC implementation uses a histogram fallback suitable for FX spot where
  volume data may be unreliable. A full volume profile will be added in a
  future iteration.
