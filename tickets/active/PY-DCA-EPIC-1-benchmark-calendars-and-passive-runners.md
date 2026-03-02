## Title
- PY-DCA-EPIC-1 — Benchmarks calendaires et runners passifs DCA

## Ticket type
- Type B: Implementation

## BMAD Stage
- Architect

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-DCA-GRID-001
- Repo Owner: quant-python-engine
- Upstream Dependencies: None
- Contract Version: dca-grid-process-v1

## Goal
- Implémenter un socle de benchmarks DCA passifs (mensuel fixe, randomisé, mid-month, turn-of-month, hebdo fixe) avec sorties homogènes et déterministes.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/strategies/runner.py`
  - `src/quant_engine/io/` (artefacts run)
  - `src/quant_engine/validate/splitter.py` (fenêtres temporelles)
  - `specs/strategy_dca_*`
- Pipeline integration points:
  - Exécution locale via CLI `qe run-local`
  - Exécution API via `/submit`
- Related docs:
  - `docs/dca_grid_implementation_process.md`
  - `docs/canonical_runs_dca_source_of_truth_2026-02-20.md`

## BMAD Handover In
- Contrat ex-ante des règles calendaires (jours non ouvrés, fallback, fuseau).

## BMAD Handover Out
- Runners benchmark utilisables par les tickets métriques/statistiques.
- Spécifications JSON minimales de démonstration par variante.

## Context7 Decision
- Required: No
- Reason: La logique repose sur des patterns existants du repo.

## Constraints & conventions
- Préserver le comportement déterministe avec `seed` obligatoire pour la variante randomisée.
- Aucune divergence de schéma entre variantes de benchmark.

## Definition of Done
- [ ] Runners benchmark implémentés dans une interface commune.
- [ ] Specs de tests ajoutées sous `specs/tests/`.
- [ ] Validation via CLI et API sans régression des stratégies DCA existantes.
- [ ] Journalisation des paramètres calendaires effectifs.

## Implementation plan
1. Ajouter une interface commune de benchmark DCA + normalisation des sorties (cashflows, ordres, capital curve).
2. Implémenter les 5 variantes passives avec règles calendaires explicites.
3. Ajouter fixtures/tests de cas limites (février, week-end, jours fériés simulés, timezone).
4. Documenter la convention de fallback calendaire.

## Tests
- Unit tests:
  - Règles calendaires pour chaque variante.
  - Déterminisme avec seed.
- Integration tests:
  - `qe run-local --spec specs/tests/strategy_dca_equity_csv_basic.json` adapté benchmark.
- Performance sanity (if applicable):
  - Monte Carlo calendrier borné (N configurable, runtime budget).

## Validation commands
- `poetry run pytest -q tests/strategies/test_dca_benchmark_calendars.py`
- `poetry run pytest -q tests/integration/test_dca_benchmark_specs.py`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Optimisation des paramètres de grille DCA.
- UI/visualisations Angular.

## Notes / pitfalls
- Attention à la définition “jour de bourse” vs calendrier crypto 24/7.
