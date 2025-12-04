from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class StrategyRunResult:
    strategy_id: str
    run_id: str
    asset_class: str
    universe: Optional[str]
    timeframe: Optional[str]

    symbol: Optional[str]
    compared_symbol: Optional[str]

    start_ts_utc: datetime
    end_ts_utc: datetime

    win_count: int
    loss_count: int
    total_return: float
    max_drawdown: float
    average_trade: float
    average_sl: Optional[float]
    average_tp: Optional[float]
    rr_moyen: Optional[float]
    total_net_return: Optional[float]
    net_win_count: Optional[int]
    net_loss_count: Optional[int]
    average_net_trade: Optional[float]

    initial_capital: Optional[float] = None
    final_capital: Optional[float] = None
    return_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    volatility_pct: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    winrate_pct: Optional[float] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletedTrade:
    strategy_id: str
    run_id: str
    symbol: str
    asset_class: str
    side: str  # ex: "LONG"
    cycle_id: Optional[int]

    entry_time_utc: datetime
    exit_time_utc: datetime
    entry_price: float
    exit_price: float
    quantity: float

    gross_pnl: float
    gross_pnl_pct: float
    max_dd_pct: Optional[float] = None

    meta: Dict[str, Any] = field(default_factory=dict)


def to_backend_payload(run: StrategyRunResult, trades: List[CompletedTrade]) -> Dict[str, Any]:
    def _iso(dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if dt is not None else None

    run_dict = {
        "strategyId": run.strategy_id,
        "runId": run.run_id,
        "assetClass": run.asset_class,
        "universe": run.universe,
        "timeframe": run.timeframe,
        "symbol": run.symbol,
        "comparedSymbol": run.compared_symbol,
        "startTsUtc": _iso(run.start_ts_utc),
        "endTsUtc": _iso(run.end_ts_utc),
        "winCount": run.win_count,
        "lossCount": run.loss_count,
        "totalReturn": run.total_return,
        "maxDrawdown": run.max_drawdown,
        "averageTrade": run.average_trade,
        "averageSL": run.average_sl,
        "averageTP": run.average_tp,
        "rrMoyen": run.rr_moyen,
        "totalNetReturn": run.total_net_return,
        "netWinCount": run.net_win_count,
        "netLossCount": run.net_loss_count,
        "averageNetTrade": run.average_net_trade,
        "initialCapital": run.initial_capital,
        "finalCapital": run.final_capital,
        "returnPct": run.return_pct,
        "maxDrawdownPct": run.max_drawdown_pct,
        "volatilityPct": run.volatility_pct,
        "sharpe": run.sharpe,
        "sortino": run.sortino,
        "winratePct": run.winrate_pct,
        "extra": run.extra,
    }

    trade_dicts: List[Dict[str, Any]] = []
    for t in trades:
        trade_dicts.append(
            {
                "strategyId": t.strategy_id,
                "runId": t.run_id,
                "symbol": t.symbol,
                "assetClass": t.asset_class,
                "side": t.side,
                "cycleId": t.cycle_id,
                "entryTimeUtc": _iso(t.entry_time_utc),
                "exitTimeUtc": _iso(t.exit_time_utc),
                "entryPrice": t.entry_price,
                "exitPrice": t.exit_price,
                "quantity": t.quantity,
                "grossPnl": t.gross_pnl,
                "grossPnlPct": t.gross_pnl_pct,
                "maxDdPct": t.max_dd_pct,
                "meta": t.meta,
            }
        )

    return {"run": run_dict, "trades": trade_dicts}
