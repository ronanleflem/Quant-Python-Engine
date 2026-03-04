"""Contract for executable strategy implementations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    DataFrame = pd.DataFrame
else:
    DataFrame = Any


@runtime_checkable
class StrategyContract(Protocol):
    """Generate strategy decisions from historical bars and execution context.

    Inputs:
        ohlcv: Historical bars including the latest closed bar, sorted by timestamp.
        context: Runtime metadata such as open positions, risk limits, and config overrides.

    Output:
        Sequence of signal payloads ready for execution/backtest layers.
    """

    def evaluate(self, ohlcv: DataFrame, context: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        """Produce zero or more normalized signals for the current evaluation call."""
