## Title
- Qualifier les champs backtest Python supportes vs non implementes pour Strategy Launcher

## Ticket type
- Type B: Implementation

## BMAD Stage
- Done

## Status
- Done

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-002
- Repo Owner: quant-python-engine
- Upstream Dependencies:
  - `C:\Users\ronan\Desktop\Projet Finance\spring\Financial-Project\tickets\active\INIT-002-spring-strategy-launcher-backtest-not-implemented.md`
  - Coordination UI: `C:\Users\ronan\Desktop\Angular-Front-Financial\Angular-Financial-Project\tickets\active\INIT-002-angular-strategy-launcher-backtest-not-implemented.md`
- Contract Version: catalog_version=2026-02-02

## Goal
- Determiner et fiabiliser le comportement Python face au payload backtest Strategy Launcher en distinguant ce qui est implemente de ce qui ne l'est pas, pour permettre une restitution claire `Not implemented yet` cote frontend.

## Context / Entry points
- Modules/files:
  - parser/validator de payload backtest
  - modules de strategie/signal/tp_sl/filters utilises au runtime
- Pipeline integration points:
  - entree payload backtest depuis Spring
  - execution runner backtest
  - sortie erreurs/warnings metadata
- Related docs:
  - `docs/GENERATE_TICKET_FROM_JIRA.md`
  - `tickets/_templates/TICKET_TEMPLATE.md`
  - `tickets/_templates/AUDIT_PROMPT.md`

## BMAD Handover In
- Contrat cible `catalog_version=2026-02-02` cadre par INIT/CP.
- Clarification attendue de Spring sur canal de signalisation (error/warning/message).
- Scope Architect: figer la matrice supporte/non supporte et proposer une signalisation standard unique.

## BMAD Handover Out
- Matrice Python Architecture: champs supportes / non supportes (niveau contrat et niveau runtime).
- Standard de signalisation cible (codes, format, severite) pour les champs non implementes.
- Plan Dev cible (sans code ici) pour brancher la signalisation sans changement de semantique existante.

## Context7 Decision
- Required: No
- Reason: Le travail est interne au moteur Python et au contrat localement fourni, sans dependance documentaire externe.

## Constraints & conventions
- Preserve deterministic behavior unless specified otherwise.
- Prefer vectorized numpy/pandas operations.
- Avoid silent API/contract changes.
- Ne pas changer la semantique des champs deja supportes.
- Un champ non supporte ne doit pas etre traite silencieusement sans trace exploitable.

## Definition of Done
- [x] Les champs du payload fourni sont classes supporte/non supporte avec evidence code.
- [x] Une signalisation standard est definie pour les cas non supportes.
- [x] Les impacts et invariants sont explicites pour la phase Dev.
- [x] Implementation phase Dev realisee sur le scope ticket.
- [x] Tests cibles API/worker ajoutes et passants sur le scope.
- [x] Validations/tests du ticket executes et passants.

## Architecture decisions
1. Le pipeline canonique `/runs` valide et enfile la requete, mais n'execute pas encore le backtest:
   - evidence: `src/quant_engine/api/app.py:388` a `src/quant_engine/api/app.py:394`.
2. La validation d'entree est stricte (`extra="forbid"`), sauf certains sous-blocs libres (`strategy.params`, `tp_sl` en dict libre):
   - evidence: `src/quant_engine/api/run_request_input.py:11`, `src/quant_engine/api/run_request_input.py:84`, `src/quant_engine/api/run_request_input.py:115`.
3. Le moteur backtest supporte techniquement `signal`, `filters`, `screening`, `tpsl.dynamic_sl`, `tpsl.jitter` via spec interne non-canonique:
   - evidence: `src/quant_engine/backtest/runner.py:223`, `src/quant_engine/backtest/runner.py:259`, `src/quant_engine/backtest/runner.py:327`, `src/quant_engine/backtest/engine.py:138`.
4. Risque actuel: certains champs canonique sont acceptes mais non consommes (silent no-op), ce qui doit etre signale explicitement au lieu d'etre implicite.

## Matrice supporte / non supporte
| Section payload Strategy Launcher | Champ(s) | Statut contrat `/runs` | Statut runtime actuel | Decision Architect |
| --- | --- | --- | --- | --- |
| `signal` | `type`, `fast`, `slow`, `require_crossing` | Supporte (validation OK) | Non supporte (non execute en canonique) | `NON_SUPPORTE_RUNTIME` tant que mapping vers spec interne absent |
| `filters` | `filters[]`, `rules[]`, `rules_config` | Supporte (validation OK) | Non supporte (non execute en canonique) | `NON_SUPPORTE_RUNTIME` |
| `screening` (top-level) | bloc complet | Non supporte (rejete en 422, extra field) | N/A | `NON_SUPPORTE_CONTRAT` |
| `strategy.params.tp_sl` | sous-champs libres (`dynamic_sl`, `jitter`, etc.) | Supporte syntaxiquement (dict libre) | Non supporte (non consomme en canonique) | `NON_SUPPORTE_RUNTIME` |
| `strategy.name` | nom strategie | Supporte (validation OK) | Non supporte (non utilise) | `NON_SUPPORTE_RUNTIME` |

## Signalisation standard cible
- Canal 1 (contrat invalide): HTTP `422` avec format deja en place `{ "errors": [ { "field", "code", "message" } ] }`.
- Canal 2 (champ accepte mais non implemente runtime): run termine en `FAILED` avec payload:
  - `error.code = "not_implemented_feature"`
  - `error.message = "Feature not implemented for canonical backtest run"`
  - `error.details[] = [{ "field": "<payload.path>", "reason": "accepted_but_not_wired" }]`
- Canal 3 (info non bloquante): reserve pour plus tard; pas de warning silencieux tant que le frontend attend `Not implemented yet`.
- Regle d'invariance: aucun champ non supporte ne doit etre ignore sans trace exploitable cote API result (`/runs/{id}/result`) et logs.

## Mapping standard des codes
- `NON_SUPPORTE_CONTRAT` -> HTTP 422 (`code` pydantic ex: `extra_forbidden` / `missing`).
- `NON_SUPPORTE_RUNTIME` -> status run `FAILED` + `error.code = "not_implemented_feature"`.
- `MAL_CONFIGURE` (champ supporte mais valeur invalide) -> conserver les erreurs de validation/metier existantes, distinctes de `not_implemented_feature`.

## Implementation plan
1. Ajouter une etape de qualification des champs canonique backtest avant execution de job canonique.
2. Lever un resultat `FAILED` standardise quand un champ est `NON_SUPPORTE_RUNTIME` (payload error unifie).
3. Conserver `422` pour `NON_SUPPORTE_CONTRAT` sans changer le format d'erreur existant.
4. Ajouter tests API sur les cas:
   - `screening` top-level -> `422`.
   - `signal`/`filters`/`strategy.params.tp_sl` presents -> `FAILED` + `not_implemented_feature`.
5. Valider compatibilite de libelle avec Spring/Angular (`Not implemented yet`).

## Tests
- Unit tests:
  - validator canonique: separation stricte contrat vs runtime
- Integration tests:
  - cycle `/runs` -> worker -> `/runs/{id}/result` avec codes standardises
- Performance sanity (if applicable):
  - N/A en phase Architect (a executer en Dev si logique supplementaire)

## Validation commands
- `poetry run pytest -q`

## Reviewer Gate
- [x] Scope matches ticket and DoD.
- [x] Architecture constraints respected.
- [x] Tests are meaningful and pass.
- [x] No regression risk left unaddressed.

## Dev validation summary
- Cibles executees (scope ticket):
  - `poetry run pytest -q tests/test_run_request_input_models.py::test_accepts_backtest_payload_with_java_aliases tests/test_api_runs_lifecycle_endpoints.py::test_run_result_returns_not_implemented_error_details tests/test_api_worker_queue.py::test_worker_marks_canonical_backtest_not_implemented tests/test_api_runs_submit_endpoint.py::test_runs_submit_rejects_top_level_screening_field`
  - Resultat: `4 passed`.
  - `poetry run pytest -q tests/test_api_runs_lifecycle_endpoints.py tests/test_api_worker_queue.py tests/test_api_runs_submit_endpoint.py tests/test_run_request_input_models.py`
  - Resultat: `32 passed`.
- Corrections review gate appliquees:
  - Compatibilite alias Java `specType` corrigee (suppression de la cle source apres mapping vers `spec_type`).
  - Signalisation runtime etendue a `strategy.params.tpSl` en plus de `strategy.params.tp_sl`.
  - Test API ajoute sur `/runs/{id}/result` pour valider `FAILED + not_implemented_feature + details`.

## Final summary
- Changement principal: standardisation de la signalisation pour backtest canonique non implemente (`FAILED` + `not_implemented_feature` + `details` exploitables), sans changement de semantique des champs supportes.
- Corrections review gate: alias Java `specType` corrige, support de `tpSl` dans la qualification des champs non cables, couverture test API result renforcee.
- Validation finale ticket: `poetry run pytest -q tests/test_api_runs_lifecycle_endpoints.py tests/test_api_runs_submit_endpoint.py tests/test_api_worker_queue.py tests/test_run_request_input_models.py` -> `32 passed`.

## Residual risks
- Aucun risque residuel bloquant identifie sur le scope de ce ticket.

## Non-goals / Out of scope
- Ajouter l'implementation complete de toutes options non supportees.
- Refonte complete du moteur backtest.
- Changement de contrat hors `catalog_version=2026-02-02`.

## Notes / pitfalls
- Cas sensibles probables dans le payload fourni: `dynamic_sl`, `jitter`, `screening`, `filters.rules/rules_config` (a valider sur code).
- Bien distinguer "non implemente" de "mal configure" pour eviter messages trompeurs.
- Synchroniser les libelles/messages avec Spring pour affichage UI coherent.
- Point de vigilance: ne pas marquer "supporte" un champ uniquement parce qu'il passe la validation d'entree.
