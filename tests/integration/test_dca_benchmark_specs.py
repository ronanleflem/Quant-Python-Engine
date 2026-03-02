from __future__ import annotations

import sys
import types
from pathlib import Path

if "pymysql" not in sys.modules:
    pymysql_mod = types.ModuleType("pymysql")
    cursors_mod = types.ModuleType("pymysql.cursors")

    class _DictCursor:  # pragma: no cover - import shim
        pass

    cursors_mod.DictCursor = _DictCursor
    pymysql_mod.cursors = cursors_mod
    sys.modules["pymysql"] = pymysql_mod
    sys.modules["pymysql.cursors"] = cursors_mod

from quant_engine.strategies import runner as strategies_runner


SPEC_DIR = Path("specs/tests")


def test_benchmark_specs_run_and_share_output_schema(monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    names = [
        "strategy_dca_benchmark_monthly_fixed.json",
        "strategy_dca_benchmark_monthly_randomized.json",
        "strategy_dca_benchmark_mid_month.json",
        "strategy_dca_benchmark_turn_of_month.json",
        "strategy_dca_benchmark_weekly_fixed.json",
    ]
    required = {
        "variant",
        "planned_date",
        "effective_date",
        "fallback",
        "cashflow",
        "invested_capital",
        "capital_curve",
        "timezone",
    }

    for name in names:
        spec = strategies_runner.load_strategy_spec(SPEC_DIR / name)
        result = strategies_runner.run_backtest_from_spec(spec)
        signals = result["signals"]["SPY"]
        assert signals
        sample = signals[0]["meta"]["benchmark"]
        assert required.issubset(sample.keys())
