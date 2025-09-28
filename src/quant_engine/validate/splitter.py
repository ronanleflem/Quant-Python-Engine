"""Walk-forward validation utilities."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict, Any


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
