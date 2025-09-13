"""Simple optimisation runner with walk-forward evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from ..core import dataset
from ..core.spec import Spec
from ..signals.ema_cross import EmaCross
from ..core.features import atr
from ..backtest import engine
from ..validate import splitter
from ..io import artifacts


def _iterate_search_space(search_space: Dict[str, List[int]]):
    fasts = search_space.get("ema_fast", [])
    slows = search_space.get("ema_slow", [])
    for fast in fasts:
        for slow in slows:
            if fast >= slow:
                continue
            yield {"ema_fast": fast, "ema_slow": slow}


def run(spec: Spec, out_dir: str | Path = Path(".")) -> Dict[str, Any]:
    data = dataset.load_dataset(spec.data)
    val = spec.strategy.validation
    folds = splitter.generate_folds(
        data, val.train_months, val.test_months, val.folds, val.embargo_days
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    best_folds: List[Any] = []

    for params in _iterate_search_space(spec.strategy.search_space):
        for r in range(1, 6):
            fold_metrics = []
            fold_artifacts = []
            for fold in folds:
                test = fold["test"]
                signal = EmaCross(params["ema_fast"], params["ema_slow"]).generate(test)
                atr_vals = atr.compute(test)
                trades, equity, summary = engine.run(
                    test,
                    signal,
                    atr_vals,
                    spec.strategy.tpsl.atr_k,
                    r,
                )
                fold_metrics.append(summary)
                fold_artifacts.append((trades, equity))
            if not fold_metrics:
                continue
            avg_sharpe = sum(m["sharpe"] for m in fold_metrics) / len(fold_metrics)
            trial = {"params": {**params, "R": r}, "metrics": {"sharpe": avg_sharpe}}
            trials.append(trial)
            if (
                len(fold_artifacts[0][0]) >= val.min_trades
                and (best is None or avg_sharpe > best["metrics"]["sharpe"])
            ):
                best = trial
                best_folds = fold_artifacts

    artifacts.write_trials(out_dir / "trials.parquet", [
        {**t["params"], **t["metrics"]} for t in trials
    ])

    result: Dict[str, Any] = {"trials_path": str(out_dir / "trials.parquet"), "trials": trials}
    if best is not None:
        summary = {"params": best["params"], "metrics": best["metrics"]}
        artifacts.write_summary(out_dir / "summary.json", summary)
        trades_paths: List[str] = []
        equity_paths: List[str] = []
        for idx, (trades, equity) in enumerate(best_folds):
            t_path = out_dir / f"trades_{idx}.parquet"
            e_path = out_dir / f"equity_{idx}.parquet"
            artifacts.write_trades(t_path, trades)
            artifacts.write_equity(e_path, equity)
            trades_paths.append(str(t_path))
            equity_paths.append(str(e_path))
        result["best"] = {
            "params": best["params"],
            "metrics": best["metrics"],
            "trades": trades_paths,
            "equity": equity_paths,
            "summary": str(out_dir / "summary.json"),
        }
    else:
        result["best"] = None
    return result
