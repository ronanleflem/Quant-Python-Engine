from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from quant_engine.api.schemas import SeasonalitySpec, StatsSpec
from quant_engine.persistence import db as qe_db
from quant_engine.seasonality import runner as seasonality_runner
from quant_engine.stats import runner as stats_runner


STATS_SPEC = Path("specs/tests/stats_seasonality_combo_stats.json")
SEAS_SPEC = Path("specs/tests/stats_seasonality_combo_seasonality.json")
STATS_PERSIST_SPEC = Path("specs/tests/stats_seasonality_combo_stats_persist.json")
SEAS_PERSIST_SPEC = Path("specs/tests/stats_seasonality_combo_seasonality_persist.json")


def test_stats_seasonality_combo(tmp_path: Path) -> None:
    stats_payload = json.loads(STATS_SPEC.read_text())
    stats_payload["artifacts"] = {"out_dir": str(tmp_path / "combo_stats")}
    stats_spec = StatsSpec.model_validate(stats_payload)
    stats_df = stats_runner.run_stats(stats_spec)
    assert not stats_df.empty

    seas_payload = json.loads(SEAS_SPEC.read_text())
    seas_payload["artifacts"] = {"out_dir": str(tmp_path / "combo_seasonality")}
    seas_spec = SeasonalitySpec.model_validate(seas_payload)
    result = seasonality_runner.run(seas_spec)
    assert "active_bins" in result
    assert set(result.get("active_bins", {}).keys()).issubset(set(seas_spec.signal.dims))


def test_stats_seasonality_persistence_shared_ids(monkeypatch, tmp_path: Path) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    qe_db.init_db(conn)

    @contextmanager
    def memory_session():
        yield conn
        conn.commit()

    monkeypatch.setattr(qe_db, "session", memory_session)
    stats_payload = json.loads(STATS_PERSIST_SPEC.read_text())
    stats_payload["artifacts"] = {"out_dir": str(tmp_path / "combo_stats_persist")}
    stats_spec = StatsSpec.model_validate(stats_payload)
    stats_df = stats_runner.run_stats(stats_spec)
    assert not stats_df.empty

    seas_payload = json.loads(SEAS_PERSIST_SPEC.read_text())
    seas_payload["artifacts"] = {"out_dir": str(tmp_path / "combo_seasonality_persist")}
    seas_spec = SeasonalitySpec.model_validate(seas_payload)
    result = seasonality_runner.run(seas_spec)
    assert result.get("run_id")

    row = conn.execute(
        "SELECT COUNT(*) AS n FROM market_stats WHERE spec_id = ? AND dataset_id = ?",
        ("COMBO_SPEC", "COMBO_DATA"),
    ).fetchone()
    assert row["n"] > 0
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM seasonality_runs WHERE spec_id = ? AND dataset_id = ?",
        ("COMBO_SPEC", "COMBO_DATA"),
    ).fetchone()
    assert row["n"] > 0
    profiles_row = conn.execute(
        "SELECT COUNT(*) AS n FROM seasonality_profiles WHERE spec_id = ? AND dataset_id = ?",
        ("COMBO_SPEC", "COMBO_DATA"),
    ).fetchone()
    if result.get("active_bins"):
        assert profiles_row["n"] > 0
