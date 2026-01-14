from __future__ import annotations

from pathlib import Path

from quant_engine.optimize import variants as optimize_variants


SPEC_DIR = Path("specs/tests")


def _assert_opt_result(result: dict, expected_trials: int | None = None) -> None:
    assert result.get("best") is not None
    assert result.get("summary")
    assert result.get("trials_path")
    if expected_trials is not None:
        assert result.get("total_trials") == expected_trials


def test_optimize_backtest_grid_basic(tmp_path) -> None:
    spec = optimize_variants.backtest_runner.load_backtest_spec(
        SPEC_DIR / "optimize_backtest_grid_basic.json"
    )
    result = optimize_variants.run_backtest_optimization(spec, out_dir=tmp_path / "opt_bt_grid")
    _assert_opt_result(result, expected_trials=8)
    best = result.get("best") or {}
    params = best.get("params") or {}
    expected_keys = set(spec.get("optimization", {}).get("search_space", {}).keys())
    assert expected_keys.issubset(params.keys())


def test_optimize_backtest_random_basic(tmp_path) -> None:
    spec = optimize_variants.backtest_runner.load_backtest_spec(
        SPEC_DIR / "optimize_backtest_random_basic.json"
    )
    result = optimize_variants.run_backtest_optimization(spec, out_dir=tmp_path / "opt_bt_rand")
    _assert_opt_result(result, expected_trials=3)


def test_optimize_strategy_grid_basic(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    spec = optimize_variants.strategy_runner.load_strategy_spec(
        SPEC_DIR / "optimize_strategy_grid_basic.json"
    )
    result = optimize_variants.run_strategy_optimization(spec, out_dir=tmp_path / "opt_strat_grid")
    _assert_opt_result(result, expected_trials=2)


def test_optimize_strategy_random_basic(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    spec = optimize_variants.strategy_runner.load_strategy_spec(
        SPEC_DIR / "optimize_strategy_random_basic.json"
    )
    result = optimize_variants.run_strategy_optimization(spec, out_dir=tmp_path / "opt_strat_rand")
    _assert_opt_result(result, expected_trials=2)


def test_optimize_strategy_random_seed_repro(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DB_DSN", raising=False)
    spec = optimize_variants.strategy_runner.load_strategy_spec(
        SPEC_DIR / "optimize_strategy_random_basic.json"
    )
    first = optimize_variants.run_strategy_optimization(spec, out_dir=tmp_path / "opt_strat_seed_a")
    second = optimize_variants.run_strategy_optimization(spec, out_dir=tmp_path / "opt_strat_seed_b")
    assert (first.get("best") or {}).get("params") == (second.get("best") or {}).get("params")
