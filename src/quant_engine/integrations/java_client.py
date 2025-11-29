"""HTTP client helpers to interact with the Java backend services."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

BASE_URL = os.getenv("QE_JAVA_BASE_URL", "http://localhost:8080")


def get_scan(endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch a scan payload from the Java backend using ``GET``."""

    url = BASE_URL + endpoint
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    return resp.json()


def get_market_scan(scan_type: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch one of the supported market scans exposed by the Java backend."""

    url = BASE_URL + f"/api/market/scans/{scan_type}"
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    return resp.json()


def get_ohlc(
    symbol: str, asset_class: str, start: str, end: str, timeframe: str
) -> List[Dict[str, Any]]:
    """Fetch OHLC rows from the Java backend for a given instrument."""

    url = BASE_URL + "/api/market/ohlc"
    params = {
        "symbol": symbol,
        "assetClass": asset_class,
        "start": start,
        "end": end,
        "timeframe": timeframe,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def request_historical_ingestion(
    symbol: str, asset_class: str, source: str, start: str, end: str
) -> Dict[str, Any]:
    """Trigger a historical ingestion job for missing OHLC data."""

    url = BASE_URL + "/api/market/ingestion/requestHistorical"
    payload = {
        "symbol": symbol,
        "assetClass": asset_class,
        "source": source,
        "start": start,
        "end": end,
    }
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_positions() -> List[Dict[str, Any]]:
    """Return live positions if the Java backend exposes them."""

    url = BASE_URL + "/api/portfolio/positions"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


__all__ = [
    "get_scan",
    "get_market_scan",
    "get_ohlc",
    "request_historical_ingestion",
    "get_positions",
    "BASE_URL",
]
