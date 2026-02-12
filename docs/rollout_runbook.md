# Rollout + Rollback (Python Canonical Mode)

## Objectif
Deployer progressivement le mode canonical avec rollback immediat si problemes.

## Pre-prod checklist (Python)
Variables:
`DB_DSN` ou `DB_SQLITE_PATH`
`QE_CANONICAL_MAX_ATTEMPTS` (defaut: 3)
`QE_CANONICAL_TIMEOUT_SECONDS` (optionnel)
`QE_CANONICAL_STALE_SECONDS` (optionnel)
`QE_WORKER_POLL_SECONDS` (defaut: 1)
`QE_API_BASE_URL` (si CLI)
DB accessible et migrations OK.
Worker lance separement (queue DB-backed).
Artefacts ecriture (dossiers `runs/` si utilises).
Readiness/Liveness exposes:
`GET /healthz`
`GET /readyz`
Timeouts serveurs recommandes:
API keep-alive et read timeouts >= 30s.
Worker sans timeout externe qui kill avant jobs longs.

## Commandes de verification (Python)
Liveness:
`curl.exe http://127.0.0.1:8000/healthz`
Readiness:
`curl.exe http://127.0.0.1:8000/readyz`
Submit canonical:
`curl.exe -X POST http://127.0.0.1:8000/runs -H "Content-Type: application/json" --data-binary @specs/examples/submit_canonical.json`
Status:
`curl.exe http://127.0.0.1:8000/runs/RUN_ID`
Result:
`curl.exe http://127.0.0.1:8000/runs/RUN_ID/result`
Cancel:
`curl.exe -X POST http://127.0.0.1:8000/runs/RUN_ID/cancel`

## Rollout plan (4 etapes)
1. Staging smoke test
Verifier `/healthz` et `/readyz`.
Enqueue 1 run canonical, verifier transitions `QUEUED -> RUNNING -> SUCCEEDED`.
Forcer un echec pour verifier `FAILED` + error payload.
2. Canary prod (trafic limite)
1-5% des runs en canonical.
Monitorer latence, erreurs 4xx/5xx, saturation worker.
3. Montee progressive
10% -> 25% -> 50% -> 100% par paliers.
Stabiliser chaque palier avec metrics stables.
4. Full cutover
100% canonical, legacy en standby.

## Go / No-Go (exemples)
Go:
5xx < 0.5% sur 15 min.
409 cancel attendu uniquement sur runs termines.
Temps moyen de traitement stable.
Queue backlog sous controle (pas d'accumulation continue).
No-Go / Rollback:
5xx > 1% sur 10 min.
Backlog qui augmente > 30 min.
Stuck RUNNING au-dela du timeout.

## Rollback (immediat)
Rebasculer la source de trafic vers mode legacy (cote orchestrateur).
Laisser le worker canonical finir/annuler les jobs en cours.
Verifier `GET /readyz` OK et que la queue se vide.

## Monitoring rapide
Metrics:
`GET /metrics`
Champs utilises: latency p95/p99, status classes 2xx/4xx/5xx, timeouts.
