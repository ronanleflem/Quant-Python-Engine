"""Runtime state cache for live trading."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import pandas as pd


@dataclass
class LiveState:
    """Hold per-strategy symbol state between polling cycles."""

    strategy_id: str
    symbol: str
    timeframe: str
    history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ts"]))
    last_ts_seen: Optional[pd.Timestamp] = None
    warm: bool = False
    indicator_ctx: Dict[str, object] = field(default_factory=dict)
    position_state: Dict[str, object] = field(default_factory=dict)
    emitted_hashes: Set[str] = field(default_factory=set)

    def should_process(self, ts: pd.Timestamp) -> bool:
        """Return ``True`` if the provided timestamp is new for this state."""

        if ts is None:
            return False
        cur = pd.Timestamp(ts)
        if cur.tzinfo is None:
            cur = cur.tz_localize("UTC")
        else:
            cur = cur.tz_convert("UTC")
        if self.last_ts_seen is None:
            return True
        prev = self.last_ts_seen
        if prev.tzinfo is None:
            prev = prev.tz_localize("UTC")
        else:
            prev = prev.tz_convert("UTC")
        return cur > prev

    def mark_processed(self, ts: pd.Timestamp) -> None:
        """Persist the timestamp of the last processed bar."""

        cur = pd.Timestamp(ts)
        if cur.tzinfo is None:
            cur = cur.tz_localize("UTC")
        else:
            cur = cur.tz_convert("UTC")
        self.last_ts_seen = cur

    def append_history(self, df: pd.DataFrame) -> None:
        """Append fresh bars to the in-memory history buffer."""

        if df.empty:
            return
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        if "ts" not in self.history.columns:
            self.history = df
        elif self.history.empty:
            self.history = df
        else:
            self.history = (
                pd.concat([self.history, df], ignore_index=True)
                .drop_duplicates(subset=["ts"], keep="last")
                .sort_values("ts")
                .reset_index(drop=True)
            )


__all__ = ["LiveState"]
