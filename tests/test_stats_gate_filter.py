from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from quant_engine.persistence import db as qe_db
from quant_engine.persistence.repo import MarketStatsRepository
from quant_engine.strategies import runner as strategies_runner


SPEC_PATH = Path("specs/tests/strategy_dca_equity_stats_gate.json")
SPEC_MISSING_PATH = Path("specs/tests/strategy_dca_equity_stats_gate_missing.json")
SPEC_CONDITION_PATH = Path("specs/tests/strategy_dca_equity_stats_gate_condition.json")


def test_stats_gate_filter_allows_run(monkeypatch) -> None:
    monkeypatch.setenv("DB_DSN", f"sqlite:///{Path('tests/data/stats_gate.db')}")
    spec = json.loads(SPEC_PATH.read_text())
    result = strategies_runner.run_backtest_with_payload(spec)
    payload = result.get("payload", {})
    assert "run" in payload
    assert payload["run"]["symbol"] == "SPY"


def test_stats_gate_allow_if_missing_false_raises(monkeypatch) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    qe_db.init_db(conn)

    @contextmanager
    def memory_session():
        yield conn
        conn.commit()

    monkeypatch.setattr(qe_db, "session", memory_session)
    spec = json.loads(SPEC_MISSING_PATH.read_text())
    with pytest.raises(ValueError, match="No stats found"):
        strategies_runner.run_backtest_with_payload(spec)


def test_stats_gate_condition_value_and_metric_lift(monkeypatch) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    qe_db.init_db(conn)
    repo = MarketStatsRepository(conn)
    repo.bulk_upsert(
        [
            {
                "symbol": "SPY",
                "timeframe": "1D",
                "event": "always_true",
                "condition_name": "day_of_week",
                "condition_value": "2",
                "target": "up_next_bar",
                "split": "test",
                "n": 500,
                "successes": 350,
                "p_hat": 0.7,
                "ci_low": 0.6,
                "ci_high": 0.8,
                "lift": 1.2,
                "start": "2025-01-01",
                "end": "2025-01-20",
                "spec_id": None,
                "dataset_id": None,
            }
        ]
    )

    @contextmanager
    def memory_session():
        yield conn
        conn.commit()

    monkeypatch.setattr(qe_db, "session", memory_session)
    spec = json.loads(SPEC_CONDITION_PATH.read_text())
    result = strategies_runner.run_backtest_with_payload(spec)
    payload = result.get("payload", {})
    assert payload.get("run", {}).get("symbol") == "SPY"
