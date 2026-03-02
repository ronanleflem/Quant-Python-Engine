"""Walk-forward validation utilities."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict, Any, Sequence


def _add_months(dt: datetime, months: int) -> datetime:
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, 28)  # keep things simple
    return datetime(year, month, day)


def generate_folds(
    dataset: List[Dict[str, Any]],
    train_months: int,
    test_months: int,
    folds: int,
    embargo_days: int,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    """Return a list of ``{"train": [], "test": []}`` splits."""
    if not dataset:
        return []
    dates = []
    for row in dataset:
        dt = datetime.fromisoformat(row["timestamp"])
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        dates.append(dt)
    start = dates[0]
    out: List[Dict[str, List[Dict[str, Any]]]] = []
    for f in range(folds):
        train_start = _add_months(start, f * test_months)
        train_end = _add_months(train_start, train_months)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = _add_months(test_start, test_months)
        train_rows = [r for r, d in zip(dataset, dates) if train_start <= d < train_end]
        test_rows = [r for r, d in zip(dataset, dates) if test_start <= d < test_end]
        if not test_rows:
            break
        out.append({"train": train_rows, "test": test_rows})
    return out


def generate_dca_rolling_windows(
    dataset: List[Dict[str, Any]],
    window_years: Sequence[int] = (3, 5, 10),
    step_months: int = 1,
) -> List[Dict[str, Any]]:
    """Build rolling windows for DCA analyses with temporal indexing.

    Each element contains:
    - ``window_years``
    - ``start`` / ``end`` (ISO timestamps)
    - ``index_ts``: temporal index of the window (window end)
    - ``rows``: dataset rows inside ``[start, end)``
    """
    if not dataset:
        return []

    if step_months <= 0:
        raise ValueError("step_months must be > 0")

    dates: List[datetime] = []
    for row in dataset:
        dt = datetime.fromisoformat(row["timestamp"])
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        dates.append(dt)

    start = min(dates)
    end = max(dates)
    ordered_windows = sorted({int(y) for y in window_years if int(y) > 0})
    out: List[Dict[str, Any]] = []

    for years in ordered_windows:
        cursor = start
        months = years * 12
        while cursor <= end:
            window_end = _add_months(cursor, months)
            if window_end > end:
                break
            rows = [r for r, d in zip(dataset, dates) if cursor <= d < window_end]
            out.append(
                {
                    "window_years": years,
                    "start": cursor.isoformat(),
                    "end": window_end.isoformat(),
                    "index_ts": window_end.isoformat(),
                    "rows": rows,
                }
            )
            cursor = _add_months(cursor, step_months)
    return out
