from __future__ import annotations

import json
from pathlib import Path

from quant_engine.strategies import runner as strategies_runner


SPEC_PATH = Path("specs/tests/strategy_dca_equity_stats_gate.json")


def test_stats_gate_filter_allows_run(monkeypatch) -> None:
    monkeypatch.setenv("DB_DSN", f"sqlite:///{Path('tests/data/stats_gate.db')}")
    spec = json.loads(SPEC_PATH.read_text())
    result = strategies_runner.run_backtest_with_payload(spec)
    payload = result.get("payload", {})
    assert "run" in payload
    assert payload["run"]["symbol"] == "SPY"
