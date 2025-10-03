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
* **POC** – simplified point-of-control using histogram counts.

Round numbers (RN) can also be generated statically for convenience.

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
