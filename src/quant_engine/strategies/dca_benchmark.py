from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from .base import Strategy, StrategySignal

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SchedulePoint:
    planned: pd.Timestamp
    effective: pd.Timestamp
    fallback: str


class DcaBenchmarkStrategy(Strategy):
    """Passive DCA benchmark runner with deterministic calendar variants."""

    SUPPORTED_VARIANTS = {
        "monthly_fixed",
        "monthly_randomized",
        "mid_month",
        "turn_of_month",
        "weekly_fixed",
    }

    def __init__(self, strategy_id: str, params: Dict[str, Any]) -> None:
        self.strategy_id = strategy_id
        self.params = params or {}
        self.asset_class = str(self.params.get("asset_class", "EQUITY")).upper()
        self.variant = str(self.params.get("variant", "monthly_fixed")).strip().lower()
        if self.variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported benchmark variant: {self.variant}")
        self.amount = float(self.params.get("amount", 100.0))
        if self.amount <= 0:
            raise ValueError("amount must be > 0")
        self.timezone = str(self.params.get("timezone", "UTC"))
        self.fallback = str(self.params.get("fallback", "next_business_day")).lower()
        if self.fallback not in {"next_business_day", "previous_business_day", "skip"}:
            raise ValueError("fallback must be next_business_day, previous_business_day or skip")
        self.holiday_dates = {
            pd.Timestamp(d).tz_localize(None).normalize() for d in self.params.get("holiday_dates", [])
        }

        if self.variant == "monthly_randomized" and self.params.get("seed") is None:
            raise ValueError("monthly_randomized benchmark requires a seed")
        self.seed = self.params.get("seed")

    def backtest(self, ohlc: pd.DataFrame, context: Dict[str, Any]) -> List[StrategySignal]:
        df = self._prepare_df(ohlc)
        symbol = str(context.get("symbol", ""))
        schedule = self._build_schedule(df)

        signals: List[StrategySignal] = []
        invested = 0.0
        units = 0.0
        for i, point in enumerate(schedule, start=1):
            row = df.loc[point.effective]
            price = float(row["close"])
            qty = self.amount / price if price > 0 else 0.0
            invested += self.amount
            units += qty
            equity = units * price
            meta = {
                "action": "buy",
                "benchmark": {
                    "variant": self.variant,
                    "order_index": i,
                    "planned_date": point.planned.isoformat(),
                    "effective_date": point.effective.isoformat(),
                    "fallback": point.fallback,
                    "cashflow": -self.amount,
                    "invested_capital": invested,
                    "capital_curve": equity,
                    "timezone": self.timezone,
                },
            }
            signals.append(
                StrategySignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    asset_class=self.asset_class,
                    side="BUY",
                    ts_open_utc=pd.Timestamp(point.effective).tz_convert("UTC"),
                    qty=qty,
                    meta=meta,
                )
            )

        LOGGER.info(
            "DCA benchmark schedule | symbol=%s variant=%s timezone=%s fallback=%s points=%d holidays=%d",
            symbol,
            self.variant,
            self.timezone,
            self.fallback,
            len(schedule),
            len(self.holiday_dates),
        )
        return signals

    def evaluate_live_bar(self, ohlc: pd.DataFrame, context: Dict[str, Any]) -> List[StrategySignal]:
        return self.backtest(ohlc, context)

    def _prepare_df(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        df = ohlc.copy()
        if "ts" in df.columns:
            idx = pd.to_datetime(df["ts"], utc=True)
            df = df.set_index(idx)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("OHLC data must provide ts column or DatetimeIndex")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()

    def _build_schedule(self, df: pd.DataFrame) -> List[_SchedulePoint]:
        local_days = pd.DatetimeIndex(df.index.tz_convert(self.timezone).normalize().unique()).sort_values()
        data_days = {d.tz_localize(None) for d in local_days}
        if not data_days:
            return []

        start = min(data_days)
        end = max(data_days)
        month_starts = pd.date_range(start=start.replace(day=1), end=end, freq="MS")

        planned_days: List[pd.Timestamp] = []
        if self.variant == "monthly_fixed":
            day = int(self.params.get("day_of_month", 1))
            for ms in month_starts:
                month_end = (ms + pd.offsets.MonthEnd(0)).day
                planned_days.append(ms.replace(day=min(day, month_end)))
        elif self.variant == "monthly_randomized":
            rng = random.Random(int(self.seed))
            for ms in month_starts:
                month_end = ms + pd.offsets.MonthEnd(0)
                candidates = [d for d in pd.date_range(ms, month_end, freq="D") if self._is_business_day(d)]
                if candidates:
                    planned_days.append(rng.choice(candidates))
        elif self.variant == "mid_month":
            start_day = int(self.params.get("start_day", 10))
            end_day = int(self.params.get("end_day", 20))
            target_day = int(self.params.get("day_of_month", (start_day + end_day) // 2))
            for ms in month_starts:
                month_end = (ms + pd.offsets.MonthEnd(0)).day
                day = min(max(target_day, start_day), end_day, month_end)
                planned_days.append(ms.replace(day=day))
        elif self.variant == "turn_of_month":
            offset = int(self.params.get("offset", -1))
            for ms in month_starts:
                month_end = ms + pd.offsets.MonthEnd(0)
                planned_days.append(month_end + pd.Timedelta(days=offset))
        elif self.variant == "weekly_fixed":
            weekday = int(self.params.get("weekday", 0))
            all_days = pd.date_range(start=start, end=end, freq="D")
            planned_days = [d for d in all_days if int(d.weekday()) == weekday]

        schedule: List[_SchedulePoint] = []
        for planned in sorted(set(pd.Timestamp(d).normalize() for d in planned_days)):
            resolved = self._resolve_execution_day(planned, data_days)
            if resolved is None:
                continue
            fallback = "none" if resolved == planned else self.fallback
            local_ts = pd.Timestamp(resolved).tz_localize(self.timezone)
            effective_utc = local_ts.tz_convert("UTC")
            schedule.append(_SchedulePoint(planned=planned, effective=effective_utc, fallback=fallback))
        return schedule

    def _resolve_execution_day(self, planned: pd.Timestamp, data_days: set[pd.Timestamp]) -> pd.Timestamp | None:
        day = pd.Timestamp(planned).normalize()
        if day in data_days and self._is_business_day(day):
            return day
        if self.fallback == "skip":
            return None

        step = 1 if self.fallback == "next_business_day" else -1
        probe = day
        for _ in range(35):
            probe = probe + pd.Timedelta(days=step)
            if probe in data_days and self._is_business_day(probe):
                return probe
        return None

    def _is_business_day(self, day: pd.Timestamp) -> bool:
        ts = pd.Timestamp(day).normalize()
        if ts.weekday() >= 5:
            return False
        return ts not in self.holiday_dates
