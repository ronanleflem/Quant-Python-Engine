## Title
- PY-DCA-EPIC-6 — Contrat de données versionné et exports reporting

## Ticket type
- Type B: Implementation

## BMAD Stage
- Architect

## Cross-Repo Coordination
- Cross-Repo Initiative: INIT-DCA-GRID-001
- Repo Owner: quant-python-engine
- Upstream Dependencies: PY-DCA-EPIC-1..5
- Contract Version: dca-grid-process-v1

## Goal
- Stabiliser un contrat de données (`schema_version`) pour alimenter Angular et exports synthèse (JSON/Parquet) reproductibles.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/io/`
  - `src/quant_engine/api/app.py`
  - `docs/spec_builders_reference.md`

## Definition of Done
- [ ] Schémas de sortie documentés et versionnés.
- [ ] Artefacts: metrics, distributions, capital_curves, rolling, score.
- [ ] Métadonnées de reproductibilité (seed, hash dataset, config version).
- [ ] Backward compatibility policy définie.

## Implementation plan
1. Définir schémas Pydantic/dataclasses pour les artefacts DCA.
2. Implémenter writers JSON/Parquet standardisés.
3. Ajouter endpoint/listing artefacts de run.

## Tests
- Unit tests:
  - validation de schéma.
- Integration tests:
  - run complet puis lecture artefacts.

## Validation commands
- `poetry run pytest -q tests/io/test_dca_artifact_contract.py`
- `poetry run pytest -q tests/api/test_dca_artifact_endpoints.py`

## Reviewer Gate
- [ ] Scope matches ticket and DoD.
- [ ] Architecture constraints respected.
- [ ] Tests are meaningful and pass.
- [ ] No regression risk left unaddressed.

## Non-goals / Out of scope
- Implémentation Angular.
