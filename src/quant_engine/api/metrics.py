"""Minimal in-memory metrics for API endpoints."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct * (len(ordered) - 1)))))
    return ordered[k]


class MetricsStore:
    def __init__(self, *, max_samples: int = 2000) -> None:
        self._lock = threading.Lock()
        self._latency_ms: Dict[Tuple[str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self._counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._timeouts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._status_class: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._start = time.time()

    def record(
        self,
        *,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        key = (endpoint, method)
        status_class = f"{status_code // 100}xx"
        with self._lock:
            self._counts[key] += 1
            self._status_class[(endpoint, method, status_class)] += 1
            self._latency_ms[key].append(duration_ms)
            if status_code in {408, 504}:
                self._timeouts[key] += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = {
                f"{ep}:{method}": list(values)
                for (ep, method), values in self._latency_ms.items()
            }
            counts = {
                f"{ep}:{method}": count
                for (ep, method), count in self._counts.items()
            }
            timeouts = {
                f"{ep}:{method}": count
                for (ep, method), count in self._timeouts.items()
            }
            status_classes = {
                f"{ep}:{method}:{cls}": count
                for (ep, method, cls), count in self._status_class.items()
            }
        percentiles = {}
        for key, values in lat.items():
            percentiles[key] = {
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
            }
        return {
            "since": self._start,
            "counts": counts,
            "status_classes": status_classes,
            "timeouts": timeouts,
            "latency": percentiles,
        }


METRICS = MetricsStore()
