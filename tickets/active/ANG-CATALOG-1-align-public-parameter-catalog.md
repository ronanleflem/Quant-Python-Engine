## Title
- ANG-CATALOG-1 - Align public parameter_catalog with canonical market_stats/seasonality payload

## Ticket type
- Type B: Implementation

## BMAD Stage
- Done

## Status
- Done (catalog alignment complete)

## Cross-Repo Coordination
- Cross-Repo Initiative: N/A
- Repo Owner: angular-financial-project
- Upstream Dependencies: Python `/runs` + `/runs/capabilities`
- Contract Version: catalog_version=2026-02-02

## Goal
- Ensure frontend builder uses canonical input schema (not internal engine schema) for market_stats and seasonality.

## Context / Entry points
- Modules/files:
  - `public/parameter_catalog.json`
  - catalog loading service in Angular app
- Pipeline integration points:
  - spec builder pages for market_stats and seasonality

## Definition of Done
- [x] `public/parameter_catalog.json` exposes canonical fields:
  - market_stats: `data.start_date/end_date`, `stats.event|condition|target`
  - seasonality: `data.start_date/end_date`, `seasonality.profile|signal|compute|execution|risk|tp_sl`
- [x] Catalog contains `stats_expanded` block with required params for key IDs.
- [x] App consumes `public/parameter_catalog.json` (not `docs/...`).

## Implementation plan
1. Keep `public/parameter_catalog.json` as runtime source of truth.
2. Follow-up ticket: add drift guard (`docs` vs `public`) and remove legacy duplicate file.
3. Smoke test market_stats and seasonality builders with live payload preview.

## Validation evidence
- Runtime catalog loader points to `/parameter_catalog.json`:
  - `C:\Users\ronan\Desktop\Angular-Front-Financial\Angular-Financial-Project\src\app\services\parameter-catalog.service.ts`
- `public/parameter_catalog.json` includes:
  - canonical market stats/seasonality fields
  - `stats_expanded`

## Non-goals / Out of scope
- Backend validation changes.
