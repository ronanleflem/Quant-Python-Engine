# Incident runbook (Python API)

## Symptomes courants
- Pic 5xx sur `/runs` ou `/runs/{id}/result`
- Latence p95/p99 elevee
- Backlog worker qui grossit
- Timeouts frequents
- Cancel qui ne passe pas en CANCELED

## Checks rapides
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- Verifier DB dispo + espace disque
- Verifier worker actif

## Commandes de diag
- `curl.exe http://127.0.0.1:8000/metrics`
- `curl.exe http://127.0.0.1:8000/readyz`
- Queue count (SQLite):
  - `sqlite3 .db/quant.db "select status, count(*) from api_jobs group by status;"`

## Mitigation
- Reduire trafic canonical (cote orchestrateur)
- Augmenter workers si backlog
- Forcer cancel des runs en cours si besoin
- Purger jobs stuck RUNNING (requeue via stale timeout)

## Rollback
- Basculer trafic vers legacy
- Stopper worker canonical si surcharge
- Verifier queue se vide et `/readyz` OK
