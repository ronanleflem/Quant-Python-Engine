from __future__ import annotations

import importlib
import sys
import types

pymysql_module = types.ModuleType("pymysql")
cursors_module = types.ModuleType("pymysql.cursors")
cursors_module.DictCursor = object
pymysql_module.cursors = cursors_module
sys.modules.setdefault("pymysql", pymysql_module)
sys.modules.setdefault("pymysql.cursors", cursors_module)


def test_consumer_layers_can_import_core_packages() -> None:
    backtest_runner = importlib.import_module("quant_engine.backtest.runner")
    strategies_runner = importlib.import_module("quant_engine.strategies.runner")

    core_domain = importlib.import_module("quant_engine.core.domain")
    core_contracts = importlib.import_module("quant_engine.core.contracts")

    assert backtest_runner is not None
    assert strategies_runner is not None
    assert hasattr(core_domain, "Candle")
    assert hasattr(core_contracts, "MarketIntelligenceService")
