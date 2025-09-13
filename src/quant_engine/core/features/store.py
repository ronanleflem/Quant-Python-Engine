"""Feature store with a very small on-disk cache.

The cache key is built from the dataset content and the function
parameters.  Results are stored as JSON for maximal portability.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Any, List

CACHE_DIR = Path(".feature_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _hash_dataset(dataset: List[Dict[str, Any]]) -> str:
    raw = json.dumps(dataset, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()


def _hash_params(params: Dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()


def get_or_compute(
    name: str,
    dataset: List[Dict[str, Any]],
    params: Dict[str, Any],
    func: Callable[[List[Dict[str, Any]], Dict[str, Any]], List[float]],
) -> List[float]:
    """Return cached feature values or compute them."""
    ds_hash = _hash_dataset(dataset)
    param_hash = _hash_params(params)
    path = CACHE_DIR / f"{name}_{ds_hash}_{param_hash}.json"
    if path.exists():
        return json.loads(path.read_text())
    values = func(dataset, params)
    path.write_text(json.dumps(values))
    return values

