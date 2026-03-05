## Title
- PY-MI-EPIC-0 — Roadmap d’implémentation Market Intelligence (index)

## Ticket type
- Type B: Implementation

## BMAD Stage
- Architect

## Cross-Repo Coordination
- Cross-Repo Initiative: N/A
- Repo Owner: quant-python-engine
- Upstream Dependencies: docs/architecture_market_intelligence_refactor.md
- Contract Version: mi-arch-v1

## Goal
- Fournir un plan exécutable par agents Codex pour introduire la couche `market_intelligence` sans casser les flux backtest/DCA/stress/perf/stats/seasonality/optimize.

## Epic order (obligatoire)
1. PY-MI-EPIC-1 — Core contracts + import guards
2. PY-MI-EPIC-2 — Feature store + service Market Intelligence
3. PY-MI-EPIC-3 — Intégration moteurs (backtest + DCA + stress)
4. PY-MI-EPIC-4 — Analytics (performance/stats/seasonality) + alignement optimize
5. PY-MI-EPIC-5 — Migration legacy + dépréciations + hardening

## Delivery policy (Codex)
- PRs petites (200-500 LOC si possible), un seul sujet par PR.
- Chaque PR doit inclure:
  - code,
  - tests unitaires et/ou intégration,
  - note de compatibilité rétro,
  - update doc minimale.
- Interdiction de déplacer massivement les fichiers en une seule PR.

## Definition of Done
- [ ] Les 5 epics sont livrées dans l’ordre.
- [ ] Les règles d’import interdites sont testées automatiquement.
- [ ] Les runs de référence backtest + DCA restent compatibles (non-régression KPI).

## Non-goals / Out of scope
- Refonte UI ou dashboards.
- Changement des specs JSON externes non versionné.
