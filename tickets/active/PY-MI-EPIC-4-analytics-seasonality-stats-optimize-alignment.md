## Title
- PY-MI-EPIC-4 — Segmentation analytics + alignement stats/seasonality/optimize

## Ticket type
- Type B: Implementation

## BMAD Stage
- Dev

## Goal
- Aligner les capacités analytiques autour des labels MI, sans casser les modules existants.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/performance/`
  - `src/quant_engine/stats/`
  - `src/quant_engine/seasonality/`
  - `src/quant_engine/optimize/`

## Tickets enfants
- PY-MI-4.1: Ajouter segmentation performance par `regime` et `magnet_failure`.
- PY-MI-4.2: Déplacer la partie "feature computation" de `stats` vers MI (avec wrappers compat).
- PY-MI-4.3: Permettre à `seasonality` de segmenter par labels MI (optionnel via flag).
- PY-MI-4.4: Clarifier `optimize` comme orchestrateur backtest + DCA consommant les mêmes contrats MI.
- PY-MI-4.5: Snapshot tests payload analytics/stats/seasonality.

## Definition of Done
- [ ] Les rapports performance exposent des vues par régime/flag.
- [ ] `stats` conserve l’inférence, MI centralise les features de contexte.
- [ ] `seasonality` et `optimize` restent compatibles.

## Validation commands
- `poetry run pytest -q tests/stats tests/integration tests/io`

## Notes / pitfalls
- Respecter les contrats existants côté API/persistence.
