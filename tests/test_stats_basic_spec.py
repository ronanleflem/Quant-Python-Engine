from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from quant_engine.api.schemas import StatsSpec
from quant_engine.persistence import db as qe_db
from quant_engine.stats import runner as stats_runner


SPEC_PATH = Path("specs/tests/stats_basic.json")
FOLDS_SPEC_PATH = Path("specs/tests/stats_validation_folds.json")
PERSIST_SPEC_PATH = Path("specs/tests/stats_basic_persist.json")


def test_stats_basic_from_spec(tmp_path: Path) -> None:
    payload = json.loads(SPEC_PATH.read_text())
    payload["artifacts"] = {"out_dir": str(tmp_path / "stats_basic")}
    spec = StatsSpec.model_validate(payload)
    df = stats_runner.run_stats(spec)
    assert not df.empty
    assert "lift_freq" in df.columns


def test_stats_validation_folds_and_qvalues(tmp_path: Path) -> None:
    payload = json.loads(FOLDS_SPEC_PATH.read_text())
    payload["artifacts"] = {"out_dir": str(tmp_path / "stats_folds")}
    spec = StatsSpec.model_validate(payload)
    df = stats_runner.run_stats(spec)
    assert not df.empty
    assert set(df["split"]) == {"train", "test"}
    assert df["q_value"].notna().any()
    assert df["significant"].isin([True, False]).all()


def test_stats_persistence_sqlite_memory(monkeypatch, tmp_path: Path) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    qe_db.init_db(conn)

    @contextmanager
    def memory_session():
        yield conn
        conn.commit()

    monkeypatch.setattr(qe_db, "session", memory_session)
    payload = json.loads(PERSIST_SPEC_PATH.read_text())
    payload["artifacts"] = {"out_dir": str(tmp_path / "stats_persist")}
    spec = StatsSpec.model_validate(payload)
    df = stats_runner.run_stats(spec)
    assert not df.empty
    cur = conn.execute(
        "SELECT COUNT(*) AS n, MIN(spec_id) AS spec_id, MIN(dataset_id) AS dataset_id, "
        "MIN(lift_freq) AS lift_freq, MIN(lift_bayes) AS lift_bayes "
        "FROM market_stats"
    )
    row = cur.fetchone()
    assert row["n"] > 0
    assert row["spec_id"] == "STAT_MEM"
    assert row["dataset_id"] == "DATA_MEM"
    assert row["lift_freq"] is not None
    assert row["lift_bayes"] is not None
