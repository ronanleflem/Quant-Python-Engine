## Title
- PY-DATA-1 - Add canonical data coherence guardrails (asset_class/currency/symbol)

## Ticket type
- Type A: Audit/Discovery

## BMAD Stage
- Done

## Status
- Done

## Cross-Repo Coordination
- Cross-Repo Initiative: N/A
- Repo Owner: quant-python-engine
- Upstream Dependencies: Angular symbol picker
- Contract Version: catalog_version=2026-02-02

## Goal
- Define deterministic rules for symbol-resolution coherence across backtest/dca/market_stats/seasonality to avoid silent mismatches.

## Context / Entry points
- Modules/files:
  - `src/quant_engine/api/app.py`
  - `src/quant_engine/core/dataset.py`
  - `src/quant_engine/strategies/runner.py`
- Pipeline integration points:
  - canonical data mapping and delta/mysql source resolution

## Context7 Decision
- Required: No
- Reason: internal behavior audit.

## Definition of Done
- [x] Decision doc with enforce/warn matrix by spec_type.
- [x] Proposed error/warning codes and exact payload paths.
- [x] Rollout plan (warn -> enforce) with compatibility notes.

## Implementation plan
1. Audit current resolution behavior for CRYPTO/FOREX/EQUITY with symbol+currency combinations.
2. Define policy table for accepted vs rejected combinations.
3. Propose migration strategy and tests to implement in follow-up ticket.

## Current behavior audit (observed)
- Canonical symbol resolution:
  - `market_stats` / `seasonality`: `data.symbols` has priority over `data.symbol` (`_resolve_canonical_symbols` in `src/quant_engine/api/app.py`).
  - `dca`: `universe[]` has priority over `data.symbol` fallback.
- Delta resolution:
  - `asset_class` drives Delta folder mapping (notably `EQUITY|ACTION -> STOCK`) in `src/quant_engine/strategies/runner.py`.
  - if `asset_class` missing and no explicit `delta_asset_dir`, Delta source is skipped in strategy fetch path.
  - `currency` (or `delta_quotes`) influences quote path; when missing, broad defaults are used (`EUR,USD,USDT,USDC`) which can hide incoherent inputs.
- No strict canonical coherence check today:
  - Python accepts inconsistent combinations (example: symbol `BTC`, `asset_class=EQUITY`, `currency=USDT`) and attempts resolution.
  - Failures are currently late (runtime data lookup / coverage), not early contract errors.

## Decision matrix (target policy)
| Spec type | Field combo | Policy now | Target policy |
| --- | --- | --- | --- |
| backtest | `data.symbol` + `strategy.params.asset_class` + `data.currency` | accepted | warn in phase 1, enforce in phase 2 |
| dca | `universe[*].symbol` + `universe[*].asset_class` + `universe[*].currency` | accepted | enforce per universe item in phase 2 |
| market_stats | `data.symbol(s)` + `data.asset_class` + `data.currency` | accepted | warn in phase 1, enforce in phase 2 |
| seasonality | `data.symbol(s)` + `data.asset_class` + `data.currency` | accepted | warn in phase 1, enforce in phase 2 |

## Proposed validation/warning rules
1. `asset_class` SHOULD be provided for all canonical run types.
2. `currency` SHOULD be provided for all canonical run types.
3. For `asset_class=CRYPTO`, `currency` MUST be one of known quote assets (configurable allowlist; default includes `USDT`, `USDC`, `USD`, `EUR`).
4. For `asset_class=FOREX`, symbol SHOULD be base-only (`EUR`) or pair (`EURUSD` / `EUR/USD`) according to resolver mode; mixed forms should be normalized once (not silently retried everywhere).
5. For `asset_class=EQUITY|ETF`, currency SHOULD default to `USD` if omitted (warn), and non-standard quote should warn.

## Proposed error/warning contract
- Phase 1 (warn only):
  - Structured log event:
    - `event: "canonical_data_coherence_warning"`
    - `code: "data_incoherent"`
    - `details[]: { field, reason, value }`
  - Continue execution.
- Phase 2 (enforce):
  - HTTP 422 via `ApiValidationException`:
    - `field`: canonical path (examples below)
    - `code`: one of:
      - `asset_class_required`
      - `currency_required`
      - `asset_class_currency_mismatch`
      - `symbol_asset_class_mismatch`
      - `symbol_format_invalid_for_asset_class`
    - `message`: explicit constraint.

## Field path mapping for validation errors
- backtest:
  - `backtest.strategy.params.asset_class`
  - `backtest.data.currency`
  - `backtest.data.symbol`
- dca:
  - `dca.data.currency`
  - `dca.universe.{i}.asset_class`
  - `dca.universe.{i}.currency`
  - `dca.universe.{i}.symbol`
- market_stats:
  - `market_stats.data.asset_class`
  - `market_stats.data.currency`
  - `market_stats.data.symbol` / `market_stats.data.symbols`
- seasonality:
  - `seasonality.data.asset_class`
  - `seasonality.data.currency`
  - `seasonality.data.symbol` / `seasonality.data.symbols`

## Rollout plan (warn -> enforce)
1. Phase 1 (non-breaking):
  - add warning logs + capabilities note.
  - no behavior change in resolution chain.
2. Phase 2 (breaking with deprecation window):
  - enforce required fields and coherence in `/runs` validation.
  - add 422 tests for each spec type.
3. Phase 3:
  - remove fallback ambiguity (especially quote defaults for crypto).
  - optionally introduce explicit symbol resolver mode (`base_only`, `pair`, `auto`).

## Follow-up implementation ticket(s)
- `PY-DATA-2` (to create): implement phase-1 warnings + capabilities note.
- `PY-DATA-3` (to create): enforce 422 coherence checks and tests.

## Non-goals / Out of scope
- Implementing the guardrails in this ticket.
