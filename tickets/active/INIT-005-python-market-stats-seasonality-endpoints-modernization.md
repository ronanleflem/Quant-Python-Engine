# INIT-005 - Python canonical data market stats/saisonnalite

## Title
- Fiabiliser la source de donnees market stats/saisonnalite pour consommation Spring

## Ticket type
- Type B: Implementation

## BMAD Stage
- Architect

## Status
- Architect Done

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-005
- Repo Owner: quant-python-engine
- Upstream Dependencies: None
- Contract Version: MARKET-STATS-SEASONALITY-V1-2026-03-05

## Goal
- Fournir une source Python stable et testee pour market stats/saisonnalite afin d'eliminer l'usage d'anciennes data/mock cote Spring et Angular.

## Context / Entry points
- Modules/files:
  - modules stats/saisonnalite et aggregation
  - interfaces de sortie consommees par Spring
- Pipeline integration points:
  - generation des donnees stats/saisonnalite
  - exposition pour consommation backend
- Related docs:
  - docs/GENERATE_TICKET_FROM_JIRA.md
  - tickets/_templates/TICKET_TEMPLATE.md
  - tickets/_templates/AUDIT_PROMPT.md

## BMAD Handover In
- Besoins DTO/endpoints cibles confirms cote Spring/Angular.
- Version contractuelle INIT-005 definie.

## BMAD Handover Out
- Shape data canonical pour market stats/saisonnalite.
- Metadata explicite de qualite/completude.
- Couverture tests Python sur ce perimetre.
- Matrice supporte/non supporte + signalisation stable pour Spring/Angular.

## Context7 Decision
- Required: No
- Reason: implementation interne, pas d'API externe incertaine.

## Constraints & conventions
- Preserve deterministic behavior unless specified otherwise.
- Prefer vectorized numpy/pandas operations.
- Avoid silent API/contract changes.

## Architecture decisions (Phase Architect)
1. Endpoint authority et source de verite
- Le flux cible pour market stats/saisonnalite passe par `POST /runs` (`spec_type=market_stats|seasonality`) + lifecycle `/runs/{id}` / `/runs/{id}/result`.
- `GET /runs/capabilities?spec_type=market_stats|seasonality` est la source contractuelle publiee pour le front/back.

2. Frontiere contrat vs runtime
- Contrat d'entree canonical strict (`extra="forbid"`) via `run_request_input`.
- Runtime mappe `market_stats` et `seasonality` vers specs internes (`_canonical_market_stats_to_spec`, `_canonical_seasonality_to_spec`) puis execute runners Python.

3. Constat de drift a corriger en Dev
- La matrice capabilities actuelle marque certains champs `accepted_but_not_wired` alors qu'ils sont consommes au runtime:
  - `market_stats`: `data.lookback`, `data.asset_class`, `data.currency`.
  - `seasonality`: `data.asset_class`, `data.currency`.
- Ce drift est un risque de contrat pour Spring/Angular et doit etre aligne en phase Dev.

4. Politique de signalisation cible INIT-005
- `NON_SUPPORTE_CONTRAT` -> HTTP `422` (`errors[].field/code/message`).
- `NON_SUPPORTE_RUNTIME_BLOQUANT` -> `FAILED` avec `error.code in {validation_error, execution_error}`.
- `NON_SUPPORTE_RUNTIME_NON_BLOQUANT` -> ne doit plus etre silencieux; exposer une trace explicite (capabilities et/ou warnings runtime) en phase Dev.

## Matrice supporte / non supporte + signalisation
### market_stats
| Champ / bloc | Statut contrat | Statut runtime observe | Signalisation actuelle | Decision Architect |
| --- | --- | --- | --- | --- |
| `data.symbol` / `data.symbols` / `data.timeframe` | SUPPORTE | SUPPORTE | `SUCCEEDED` si valide | Conserver |
| `data.path` / `data.dataset_path` / `data.mysql` | SUPPORTE | SUPPORTE | `SUCCEEDED` si source resolue | Conserver |
| `stats.event` / `stats.condition` / `stats.target` | SUPPORTE | SUPPORTE | `SUCCEEDED` si valide | Conserver |
| `stats.validation` | SUPPORTE | SUPPORTE (mapping vers spec.validation) | `SUCCEEDED` | Conserver |
| `data.lookback` | SUPPORTE | SUPPORTE (sert a calculer fenetre quand pas de dates explicites) | `SUCCEEDED` | Reclassifier en supporte dans capabilities |
| `data.asset_class` / `data.currency` | SUPPORTE | SUPPORTE (mapping data + `delta_quotes`) | `SUCCEEDED` | Reclassifier en supporte dans capabilities |
| `data.stats_pack` / `data.session` / `data.include_weekends` | SUPPORTE | NON_SUPPORTE_RUNTIME (ignore) | Silence runtime (aujourd'hui) | Ajouter signalisation explicite (non bloquante) |
| Champs hors modele canonical | NON_SUPPORTE_CONTRAT | N/A | `422` | Conserver |
| Parametres invalides stats (ex: contraintes numeriques) | SUPPORT_CONDITIONNEL | BLOQUANT | `422` (validation service) | Conserver |

### seasonality
| Champ / bloc | Statut contrat | Statut runtime observe | Signalisation actuelle | Decision Architect |
| --- | --- | --- | --- | --- |
| `data.symbol` / `data.symbols` / `data.timeframe` | SUPPORTE | SUPPORTE | `SUCCEEDED` si valide | Conserver |
| `data.start_date`/`end_date` ou `start_year`/`end_year` | SUPPORTE | SUPPORTE | `SUCCEEDED` si valide | Conserver |
| `data.path` / `data.dataset_path` / `data.mysql` | SUPPORTE | SUPPORTE | `SUCCEEDED` si source resolue | Conserver |
| `seasonality.profile` / `seasonality.signal` / `seasonality.compute` | SUPPORTE | SUPPORTE | `SUCCEEDED` | Conserver |
| `data.asset_class` / `data.currency` | SUPPORTE | SUPPORTE (mapping data + `delta_quotes`) | `SUCCEEDED` | Reclassifier en supporte dans capabilities |
| `data.window` | SUPPORTE | NON_SUPPORTE_RUNTIME (ignore) | Silence runtime (aujourd'hui) | Ajouter signalisation explicite (non bloquante) |
| `seasonality.execution` / `seasonality.risk` / `seasonality.tp_sl` | SUPPORTE | NON_SUPPORTE_RUNTIME (ignore) | Silence runtime (aujourd'hui) | Ajouter signalisation explicite (non bloquante) |
| Champs hors modele canonical | NON_SUPPORTE_CONTRAT | N/A | `422` | Conserver |
| Erreur mapping/runner interne | SUPPORT_CONDITIONNEL | BLOQUANT | `FAILED + execution_error` | Conserver |

## Signalisation standard (cible)
- Canal 1: Contrat invalide
  - HTTP `422`
  - Body: `{ "errors": [ { "field", "code", "message" } ] }`
- Canal 2: Runtime bloquant
  - `/runs/{id}/result` avec `status=FAILED`
  - `error.code = validation_error` (config/runtime invalide) ou `execution_error` (echec execution)
- Canal 3: Runtime non bloquant (accepted but not wired)
  - Ne pas ignorer silencieusement.
  - Cible Dev: publication explicite des champs ignores (capabilities alignee + metadata warning exploitable Spring/Angular).

## Evidence code (base Architect)
- Capacities market_stats/seasonality:
  - `src/quant_engine/api/app.py:2185`
  - `src/quant_engine/api/app.py:2187`
  - `src/quant_engine/api/app.py:2231`
- Mapping market_stats:
  - `src/quant_engine/api/app.py:1288`
  - `src/quant_engine/api/app.py:1299`
  - `src/quant_engine/api/app.py:1330`
  - `src/quant_engine/api/app.py:1373`
- Mapping seasonality:
  - `src/quant_engine/api/app.py:1458`
  - `src/quant_engine/api/app.py:1506`
- Execution worker canonical:
  - `src/quant_engine/api/app.py:1728`
  - `src/quant_engine/api/app.py:1737`
  - `src/quant_engine/api/worker.py:170`
- Validation input + contraintes market_stats:
  - `src/quant_engine/api/run_request_input.py:223`
  - `src/quant_engine/api/run_request_input.py:237`
  - `src/quant_engine/api/services/run_requests.py:70`
- Tests existants (matrice runtime):
  - `tests/test_api_runs_lifecycle_endpoints.py:337`
  - `tests/test_api_runs_lifecycle_endpoints.py:353`
  - `tests/test_api_worker_queue.py:359`
  - `tests/test_api_worker_queue.py:402`

## Definition of Done
- [x] Matrice supporte/non supporte (market_stats + seasonality) etablie avec evidence code.
- [x] Signalisation actuelle vs cible formalisee (422 / FAILED / non-bloquant explicite).
- [x] Gaps de contrat/capabilities identifies pour phase Dev.
- [ ] Implementation phase Dev realisee.
- [ ] Tests unitaires/integration ajoutes/maj et passants sur scope.
- [ ] Validation commands pass.

## Implementation plan
1. Aligner capabilities `market_stats`/`seasonality` avec le runtime reel (correction drift supporte vs accepted_but_not_wired).
2. Introduire une signalisation explicite pour champs acceptes mais ignores (non bloquante, sans rupture de contrat).
3. Verrouiller par tests API lifecycle/capabilities la matrice et la signalisation.

## Tests
- Unit tests:
  - mapping canonical market_stats/seasonality (consommation des champs supportes)
- Integration tests:
  - `/runs` -> worker -> `/runs/{id}/result` pour `market_stats` et `seasonality`
  - `/runs/capabilities?spec_type=market_stats|seasonality`
- Performance sanity (if applicable):
  - N/A en Architect

## Validation commands
- `poetry run pytest -q`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Refactor global du moteur quant.
- Nouvelles features stats non demandees.
- Changement direct des APIs Spring/Angular dans ce ticket local.

## Notes / pitfalls
- Garder determinisme des aggregations et horodatage coherent inter-runs.
- Point de vigilance principal: eviter les champs "accepted but ignored" sans trace exploitable.
