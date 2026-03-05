# INIT-003 - Audit Python de couverture calculatoire pour decommission ta4j

## Title
- Evaluer la parite fonctionnelle Python vs legacy Java ta4j

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Status
- Review-ready

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-003
- Repo Owner: quant-python-engine
- Upstream Dependencies: INIT-003-spring-decommission-ta4j-computation-to-python
- Contract Version: STRAT-CALC-DECOMMISSION-V1-2026-03-05

## Goal
- Produire une matrice fiable de couverture Python pour filtres/strategies/indicateurs historiquement portes par ta4j, afin d'orienter la decommission progressive cote Java et l'alignement UI.

## Context / Entry points
- Modules/files:
  - Moteur d'execution strategy/backtest
  - Modules signaux/filtres/indicateurs et validateurs de payload
- Pipeline integration points:
  - Entrees run provenant de Spring
  - Sorties status/error/capabilities vers Spring/Angular
- Related docs:
  - docs/GENERATE_TICKET_FROM_JIRA.md
  - tickets/_templates/TICKET_TEMPLATE.md
  - tickets/_templates/AUDIT_PROMPT.md
  - C:\Users\ronan\Desktop\Cross-repo-coordination\Cross-repo-coordination\context-packs\CP-003-decommission-ta4j-computation-to-python.md

## BMAD Handover In
- Contrat versionne `STRAT-CALC-DECOMMISSION-V1-2026-03-05`.
- Inventaire Spring des capacites legacy exposees ou encore consommees.
- Priorites fonctionnelles metier pour decommission progressive.

## BMAD Handover Out
- Matrice de parite: capability legacy -> equivalent Python / gap / obsolete.
- Standard de signalisation unifie (contrat invalide vs accepte_non_cable vs erreur execution).
- Liste des pre-requis techniques pour prise en charge complete Python.
- Plan de migration par lots avec criteres de validation.

## Context7 Decision
- Required: No
- Reason: analyse basee sur implementation Python locale et contrats internes.

## Constraints & conventions
- Preserve deterministic behavior unless specified otherwise.
- Prefer vectorized numpy/pandas operations.
- Avoid silent API/contract changes.
- Scope strictement local Python, sans changement Spring/Angular direct.

## Architecture decisions (Phase Architect)
1. Source de verite capacites
- `GET /runs/capabilities` est la source canonique pour publier `supported`, `accepted_but_not_wired`, presets `supported/not_supported`, et regles runtime par `spec_type`.

2. Frontiere contrat vs runtime
- Contrat d'entree canonical `/runs`: validation stricte (`extra="forbid"`) dans `run_request_input`.
- Runtime: qualification explicite des blocs acceptes mais non cables avant execution backtest/dca.

3. Signalisation unifiee (sans silence)
- NON_SUPPORTE_CONTRAT => HTTP `422` normalise `{errors:[{field,code,message}]}`.
- NON_SUPPORTE_RUNTIME (accepte mais non cable) => run `FAILED` + `error.code=not_implemented_feature` + `error.details[{field,reason=accepted_but_not_wired}]`.
- ERREUR_EXECUTION (champ supporte mais echec runtime) => run `FAILED` + `error.code=execution_error`.

4. Cadrage decommission ta4j
- Pas de support de classes ta4j en payload canonical (pas de DSL ta4j transporte tel quel).
- Migration guidee par matrice: ce qui est deja calcule en Python est marque `SUPPORTE`; sinon `NON_SUPPORTE_CONTRAT` ou `NON_SUPPORTE_RUNTIME`.

## Matrice supporte / non supporte + signalisation
| Domaine legacy ta4j | Capacite legacy (launcher) | Statut contrat `/runs` | Statut runtime Python | Signalisation cible |
| --- | --- | --- | --- | --- |
| Signaux backtest | `signal.type=ema_cross` | SUPPORTE | SUPPORTE (`EmaCross`) | `SUCCEEDED` |
| Signaux backtest | `signal.type != ema_cross` | ACCEPTE | NON_SUPPORTE_RUNTIME | `FAILED` + `not_implemented_feature` + detail `signal` |
| Filtres pre-trade | `filters.filters[]` + `filters.rules[]` + `filters.rules_config` | SUPPORTE | SUPPORTE (apply stack + scoring rules) | `SUCCEEDED` |
| Strategie backtest | `strategy.name` | ACCEPTE | NON_SUPPORTE_RUNTIME (non consomme) | `FAILED` + `not_implemented_feature` + detail `strategy.name` |
| TP/SL backtest | `strategy.params.tp_sl` forme interne (`atr_*`, `r_mult`, `dynamic_sl`, `jitter`, etc.) | SUPPORTE | SUPPORTE (mapping vers `tpsl` runner) | `SUCCEEDED` |
| TP/SL backtest | `strategy.params.tp_sl` preset/string non mappe | ACCEPTE | NON_SUPPORTE_RUNTIME | `FAILED` + `not_implemented_feature` + detail `strategy.params.tp_sl` |
| Strategie DCA | `strategy.type in {dca_equity,dca_etf,dca_benchmark,crypto_grid}` | SUPPORTE | SUPPORTE | `SUCCEEDED` |
| Grille DCA | preset `grid_balanced` (ou `params.grid` deja explicite) | SUPPORTE | SUPPORTE | `SUCCEEDED` |
| Grille DCA | presets legacy non cables (`grid_conservative`, `grid_aggressive`) | ACCEPTE | NON_SUPPORTE_RUNTIME | `FAILED` + `not_implemented_feature` + detail `strategy.grid` |
| TP/SL DCA | formes supportees (dict interne, canonical tp/sl/break_even/trailing percent, ou `tp_X_sl_Y`) | SUPPORTE | SUPPORTE | `SUCCEEDED` |
| TP/SL DCA | forme non reconnue | ACCEPTE | NON_SUPPORTE_RUNTIME | `FAILED` + `not_implemented_feature` + detail `strategy.params.tp_sl` |
| Indicateurs/filtres ta4j hors contrat canonical | champs legacy non modelises en canonical request | NON_SUPPORTE_CONTRAT | N/A | HTTP `422` (`extra_forbidden` ou validation associee) |

## Signalisation standard detaillee
- Canal 1: Validation contrat (synchrone API)
  - Reponse: HTTP `422`, body `{ "errors": [ {"field","code","message"} ] }`.
  - Usage: champ inconnu, type invalide, bloc requis absent, spec_type non supporte.

- Canal 2: Capacite acceptee mais non cablee (asynchrone worker)
  - Reponse via `/runs/{id}/result`: `status=FAILED`.
  - Error payload:
    - `error.code = "not_implemented_feature"`
    - `error.message = "Feature not implemented for canonical <spec_type> run"`
    - `error.details[] = [{"field":"<payload.path>","reason":"accepted_but_not_wired"}]`

- Canal 3: Echec execution technique/metier
  - Reponse via `/runs/{id}/result`: `status=FAILED`.
  - Error payload: `error.code = "execution_error"` (+ message runtime).

- Invariant de decommission
  - Aucun champ non supporte ne doit etre ignore silencieusement.
  - Tout cas non supporte doit etre classifie soit `422`, soit `FAILED/not_implemented_feature`.

## Evidence code (base Architect)
- Contrat strict canonical:
  - `src/quant_engine/api/run_request_input.py:9`
  - `src/quant_engine/api/run_request_input.py:350`
- Qualification runtime non cablee backtest/dca:
  - `src/quant_engine/api/app.py:323`
  - `src/quant_engine/api/app.py:944`
- Emission `not_implemented_feature`:
  - `src/quant_engine/api/app.py:1701`
  - `src/quant_engine/api/app.py:1716`
  - `src/quant_engine/api/worker.py:170`
- Matrice capabilities exposee:
  - `src/quant_engine/api/app.py:2185`
- Signal backtest effectivement cable:
  - `src/quant_engine/backtest/runner.py:132`
  - `src/quant_engine/backtest/runner.py:140`
- Filtres disponibles/cables:
  - `src/quant_engine/filters/__init__.py:115`
  - `src/quant_engine/filters/__init__.py:171`
  - `src/quant_engine/api/app.py:3528`
- Strategies DCA disponibles/cablees:
  - `src/quant_engine/strategies/__init__.py:12`

## Gaps bloquants pour decommission complete ta4j
- Backtest canonical ne cable qu'un signal explicite (`ema_cross`) au niveau contrat runtime.
- Presets legacy DCA `grid_conservative`/`grid_aggressive` non cables en canonical.
- Une partie des options legacy est aujourd'hui seulement visible dans `legacy_dca` capabilities (parite partielle, non transport direct en canonical).

## Migration strategy (lots)
1. Lot A - Contract hardening
- Geler la taxonomie des codes d'erreur (`422`, `not_implemented_feature`, `execution_error`) entre Python/Spring/Angular.

2. Lot B - Priorites parite calculatoire
- Etendre d'abord les gaps `NON_SUPPORTE_RUNTIME` les plus frequents cote launcher (`signal` hors ema_cross, presets grid DCA).

3. Lot C - Cleanup legacy
- Reduire progressivement les surfaces `legacy_dca.not_in_canonical` apres wiring canonical equivalent.

4. Lot D - Cutover ta4j
- Basculer les routes/flux Spring restants en pass-through Python strict une fois matrice verte sur priorites metier.

## Definition of Done
- [x] Matrice supporte/non supporte exposee explicitement dans le contrat `/runs/capabilities` pour `backtest` et `dca`.
- [x] Signalisation standard (`422` / `FAILED+not_implemented_feature` / `FAILED+execution_error`) exposee explicitement dans le contrat capabilities.
- [x] Tests API ajoutes/mis a jour pour verrouiller le contrat.
- [x] Validation commands du scope executes et passants.

## Implementation plan
1. Ajouter `support_matrix` et `signaling` dans `/runs/capabilities` pour `spec_type=backtest`.
2. Ajouter `support_matrix` et `signaling` dans `/runs/capabilities` pour `spec_type=dca`.
3. Couvrir par tests endpoint capabilities (contrat JSON).

## Tests
- Unit tests:
  - N/A (scope endpoint contract)
- Integration tests:
  - `tests/test_api_runs_lifecycle_endpoints.py` (capabilities contract)
  - `tests/test_api_worker_queue.py` (non-regression signalisation worker)
- Performance sanity (if applicable):
  - N/A

## Validation commands
- `poetry run pytest -q tests/test_api_runs_lifecycle_endpoints.py -k "capabilities or not_implemented or canonical_dca_unwired_tp_sl_error or canonical_backtest_unwired_tp_sl_error"`
- `poetry run pytest -q tests/test_api_worker_queue.py tests/test_api_runs_lifecycle_endpoints.py`

## Reviewer Gate
- [x] Scope matches ticket and DoD.
- [x] Architecture constraints respected.
- [x] Tests are meaningful and pass.
- [x] No regression risk left unaddressed.

## Non-goals / Out of scope
- Aucune implementation de nouvelles strategies/filtres dans ce ticket.
- Aucune modification directe des APIs Spring.
- Aucune adaptation UI Angular dans ce ticket.

## Dev implementation summary
- Changement code:
  - Ajout de `support_matrix` et `signaling` dans la reponse `/runs/capabilities` pour `spec_type=backtest`.
  - Ajout de `support_matrix` et `signaling` dans la reponse `/runs/capabilities` pour `spec_type=dca`.
- Fichiers modifies:
  - `src/quant_engine/api/app.py`
  - `tests/test_api_runs_lifecycle_endpoints.py`
- Couverture tests ajoutee:
  - Assertions sur la matrice/backtest + signalisation.
  - Nouveau test dedie `dca` pour matrice + signalisation.

## Dev validation summary
- Commande:
  - `poetry run pytest -q tests/test_api_runs_lifecycle_endpoints.py -k "capabilities or not_implemented or canonical_dca_unwired_tp_sl_error or canonical_backtest_unwired_tp_sl_error"`
  - Resultat: `12 passed, 19 deselected`.
- Commande:
  - `poetry run pytest -q tests/test_api_worker_queue.py tests/test_api_runs_lifecycle_endpoints.py`
  - Resultat: `58 passed`.

## Notes / pitfalls
- Certaines equivalences legacy peuvent etre partielles (meme nom, semantique differente).
- Attention aux differences de defaults et de precision numerique vs Java.
- Garder la matrice en phase avec `/runs/capabilities` a chaque livraison Dev.
