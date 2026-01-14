# Test Plan (Audit + Proposed Groups)

This doc summarizes existing tests and proposes additional groups. It also notes overlap.

## Existing tests (current groups)

1) API
   - api_http_contract, api_endpoints_extended, api_runs_stats_endpoints, api_schemas_contract, api_smoke, api_time_utils

2) Backtest core
   - backtest_engine_no_lookahead, backtest_engine_tpsl_edge_cases, backtest_metrics, backtest_minimal,
     backtest_require_crossing, backtest_summary_golden
   - backtest_data_sources (new), backtest_trade_expectations (new)

3) CLI
   - cli_smoke, cli_errors

4) Data ingestion
   - dataset_ingestion, mysql_feed_unit, mysql_feed_smoke

5) Features + signals
   - features

6) Filters
   - filters_risk_mgmt_smoke, filters_stat_prob_smoke, filters_structure_ict_smoke,
     filters_time_seasonality_smoke, filters_volatility_trend_smoke, filters_volume_profile_smoke

7) Levels
   - levels_smoke, levels_db_constraints_smoke, levels_phase2a_smoke, levels_phase2b_smoke, levels_phase15_smoke

8) Live
   - live_smoke

9) Persistence
   - persistence_api, persistence_db, persistence_repo

10) Seasonality
    - seasonality_smoke, seasonality_integration, seasonality_sessions, seasonality_profiles_compare

11) Stats
    - stats_smoke, stats_integration, stats_api_endpoints, stats_bayes_fdr, stats_estimators,
      stats_levels_integration_smoke, stats

12) Strategies
    - strategies_smoke, strategy_dca_variants (new)

13) Validation/WFA
    - validation_splitter, wfa_smoke

## Redundancy check (summary)

No hard duplication found. Some intentional overlaps:
- API: multiple files cover different endpoints and schemas.
- Backtest: minimal vs engine vs golden cover different layers.
- Filters: smoke tests split by domain; no duplicate inputs.
- Seasonality/Stats: smoke vs integration cover different scopes.
- Strategies: strategies_smoke validates base strategy logic, strategy_dca_variants validates runner + CSV source.

## Proposed test groups (advanced coverage)

1) Backtest classic - data source variants
   - Specs: backtest_csv_basic, backtest_delta_source, backtest_mysql_source, backtest_java_source
   - Status: implemented.

2) Strategy DCA - core variants
   - Specs: strategy_dca_equity_csv_basic, strategy_dca_equity_csv_filters, strategy_dca_etf_csv_basic,
     strategy_dca_equity_intracandle, strategy_dca_equity_drawdown_3m, strategy_dca_equity_require_crossing
   - Tests: tests/test_strategy_dca_variants.py, tests/test_strategy_dca_stop_loss.py
   - Status: implemented.

3) Execution/trades expected (new group)
   - Goal: assert at least one trade and basic trade integrity.
   - Specs: backtest_csv_trades, strategy_dca_equity_csv_trades
   - Tests: tests/test_backtest_trade_expectations.py, tests/test_trade_expectations_details.py
   - Status: implemented.

4) Advanced optimization - baseline
   - Specs: optimize_backtest_grid_basic, optimize_backtest_random_basic,
     optimize_strategy_grid_basic, optimize_strategy_random_basic
   - Tests: tests/test_optimize_variants_baseline.py
   - Status: implemented.

5) Advanced optimization - with filters
   - Specs: optimize_backtest_filters, optimize_strategy_filters
   - Notes: includes require_crossing variants in search space
   - Status: implemented.

6) Advanced optimization - screening / pruning
   - Specs: optimize_backtest_screening, optimize_strategy_screening
   - Notes: backtest screening includes pruning (max_drawdown_pct, min_signals_after_bars)
   - Status: implemented.

7) Stats only
   - Specs: stats_basic
   - Notes: validation folds + q_value/significant checks + sqlite memory persistence
   - Status: implemented.

8) Seasonality only
   - Specs: seasonality_basic
   - Notes: return measure + topk signals + artifacts (profiles/trades/equity)
   - Status: implemented.

9) Stats + seasonality combined
   - Specs: stats_seasonality_combo_stats, stats_seasonality_combo_seasonality
   - Notes: shared spec_id/dataset_id persistence and bins/dims coherence
   - Status: implemented.

10) Stats gate / stats-based filters
   - Specs: strategy_dca_equity_stats_gate
    - Notes: allow_if_missing=false error, conditional value + non-p_hat metric
    - Status: implemented.

11) Backtest + DCA + seasonality
   - Specs: backtest_csv_basic, strategy_dca_equity_csv_basic, seasonality_basic
   - Notes: CLI combo path + artifacts isolation
   - Status: implemented.

12) Timeframe variants
   - Specs: backtest_csv_basic, backtest_csv_h1, backtest_csv_d1
   - Notes: alias normalization + default timeframe when missing
   - Status: implemented.

13) Data edge cases
   - Specs: backtest_missing_bars, backtest_naive_timestamps
   - Notes: invalid timestamp + NaN OHLC raise, missing symbol fallback
   - Status: implemented.

14) Large dataset perf / trades (manual or slow)
   - Uses: specs/examples/data/forex/EURUSD_20250101_20250601_1min.csv
   - Tests: tests/test_large_dataset_perf.py
   - Notes: optional memory cap + throughput check + filters-on variant
   - Status: implemented (marked slow).
