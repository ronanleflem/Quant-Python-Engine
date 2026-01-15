from __future__ import annotations

from pathlib import Path

from quant_engine.backtest import runner as backtest_runner


SPEC_DIR = Path("specs/tests")


def test_backtest_mtf_anomaly_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_csv_mtf_anomaly.json")
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_benford_atr_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_csv_benford_atr.json")
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_mtf_anomaly_entropy_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_mtf_anomaly_entropy_m1_m15.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_market_manipulation_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_market_manipulation.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_market_manipulation_any_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_market_manipulation_any.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_htf_poi_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_htf_poi.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_orderflow_delta_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_orderflow_delta.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_macro_cot_oi_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_macro_cot_oi.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_lower_timeframe_confluence_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_lower_timeframe_confluence.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_psychologic_news_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_psychologic_news.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_stationarity_full_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_stationarity_full.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_volatility_extras_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(
        SPEC_DIR / "backtest_csv_volatility_extras.json"
    )
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None


def test_backtest_ict_poi_spec_runs() -> None:
    spec = backtest_runner.load_backtest_spec(SPEC_DIR / "backtest_csv_ict_poi.json")
    result = backtest_runner.run_backtest_from_spec(spec)
    assert result.get("payload") is not None
