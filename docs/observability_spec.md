# Observability spec (Python API)

## Metrics endpoints
- `GET /metrics` (JSON)
- `GET /healthz` (liveness)
- `GET /readyz` (readiness)

## Metrics content
`/metrics` retourne:
- `latency`: p95/p99 par endpoint+method
- `status_classes`: counters 2xx/4xx/5xx par endpoint+method
- `counts`: total hits par endpoint+method
- `timeouts`: count 408/504 par endpoint+method

## Logs structure (JSON)
Chaque requete log:
- `event`: `http_request`
- `path`, `method`, `status`, `duration_ms`
- `correlation_id` (si header `X-Correlation-Id` ou `X-Request-Id`)

## Coverage cible
Endpoints runs:
- `POST /runs`
- `GET /runs/{id}`
- `GET /runs/{id}/result`
- `POST /runs/{id}/cancel`

## Dashboard suggestions
Widgets:
- p95/p99 latency par endpoint runs
- Taux 5xx par endpoint runs
- Taux 4xx par endpoint runs
- Timeouts par endpoint runs
- Cancel success rate
- Result success rate (terminal vs non-terminal)

## Alerting (initial)
Voir `docs/alert_rules.md`.
