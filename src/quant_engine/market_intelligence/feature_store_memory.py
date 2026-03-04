"""In-memory feature store keyed by feature metadata."""

from __future__ import annotations

from collections.abc import Hashable
from threading import RLock
from typing import Any

FeatureStoreKey = tuple[str, str, str, str]


class InMemoryFeatureStore:
    """Simple thread-safe in-memory feature store.

    Keys are normalized tuples: ``(feature_set, symbol, timeframe, version)``.
    """

    def __init__(self) -> None:
        self._data: dict[FeatureStoreKey, Any] = {}
        self._lock = RLock()

    @staticmethod
    def _build_key(feature_set: str, symbol: str, timeframe: str, version: str) -> FeatureStoreKey:
        key: FeatureStoreKey = (feature_set, symbol, timeframe, version)
        if not all(isinstance(part, Hashable) for part in key):
            raise TypeError("All key components must be hashable")
        return key

    def get(self, feature_set: str, symbol: str, timeframe: str, version: str) -> Any:
        """Return a stored payload for the exact key.

        Raises:
            KeyError: If the key is not present.
        """

        key = self._build_key(feature_set, symbol, timeframe, version)
        with self._lock:
            return self._data[key]

    def put(
        self,
        feature_set: str,
        symbol: str,
        timeframe: str,
        version: str,
        payload: Any,
        *,
        overwrite: bool = False,
    ) -> None:
        """Persist ``payload`` for a key.

        Args:
            overwrite: When ``False``, writing an existing key raises ``KeyError``.
        """

        key = self._build_key(feature_set, symbol, timeframe, version)
        with self._lock:
            if not overwrite and key in self._data:
                raise KeyError(f"Feature key already exists: {key}")
            self._data[key] = payload

    def exists(self, feature_set: str, symbol: str, timeframe: str, version: str) -> bool:
        """Return ``True`` when a key is present in the store."""

        key = self._build_key(feature_set, symbol, timeframe, version)
        with self._lock:
            return key in self._data

    def delete(self, feature_set: str, symbol: str, timeframe: str, version: str) -> bool:
        """Delete a key.

        Returns:
            ``True`` when an entry existed and was removed, otherwise ``False``.
        """

        key = self._build_key(feature_set, symbol, timeframe, version)
        with self._lock:
            return self._data.pop(key, None) is not None
