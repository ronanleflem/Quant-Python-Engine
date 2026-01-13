"""HTTP client helpers to interact with the Java backend services."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..http_client import DEFAULT_TIMEOUT, get_shared_session, request_json

BASE_URL = os.getenv("QE_JAVA_BASE_URL", "http://localhost:8090")


def _normalize_instant(value: Optional[str]) -> Optional[str]:
    """Ensure timestamps are valid ISO instants for the Java server."""

    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return text  # leave untouched if format is already custom
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def _parse_iso_instant(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO instant string to datetime, returning None on failure."""

    if not value:
        return None
    try:
        candidate = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def get_scan(endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch a scan payload from the Java backend using ``GET``."""

    url = BASE_URL + endpoint
    session = get_shared_session()
    return request_json("GET", url, params=params, session=session)


def get_market_scan(scan_type: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch one of the supported market scans exposed by the Java backend."""

    url = BASE_URL + f"/api/market/scans/{scan_type}"
    session = get_shared_session()
    return request_json("GET", url, params=params, session=session)


def get_ohlc(
    symbol: str, asset_class: Optional[str], start: Optional[str], end: Optional[str], timeframe: Optional[str]
) -> List[Dict[str, Any]]:
    """Fetch OHLC rows from the Java backend for a given instrument."""

    url = BASE_URL + "/api/market/ohlc"
    start_iso = _normalize_instant(start)
    end_iso = _normalize_instant(end)
    base_params = {"symbol": symbol, "assetClass": asset_class, "timeframe": timeframe}

    # If both dates are present, chunk by 1-year windows to satisfy backend limits.
    start_dt = _parse_iso_instant(start_iso)
    end_dt = _parse_iso_instant(end_iso)
    if start_dt and end_dt and end_dt > start_dt:
        results: List[Dict[str, Any]] = []
        current = start_dt
        session = get_shared_session()
        while current < end_dt:
            chunk_end = min(current + timedelta(days=365), end_dt)
            params = {
                **base_params,
                "start": current.isoformat().replace("+00:00", "Z"),
                "end": chunk_end.isoformat().replace("+00:00", "Z"),
            }
            chunk = request_json(
                "GET",
                url,
                params=params,
                session=session,
                timeout=DEFAULT_TIMEOUT,
            )
            if chunk:
                results.extend(chunk)
            current = chunk_end
        return results

    # Fallback single request when dates are missing or unparsable.
    params = {**base_params, "start": start_iso, "end": end_iso}
    session = get_shared_session()
    return request_json("GET", url, params=params, session=session, timeout=DEFAULT_TIMEOUT)


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
    session = get_shared_session()
    return request_json("POST", url, json=payload, session=session, timeout=DEFAULT_TIMEOUT)


def get_positions() -> List[Dict[str, Any]]:
    """Return live positions if the Java backend exposes them."""

    url = BASE_URL + "/api/portfolio/positions"
    try:
        session = get_shared_session()
        resp = session.get(url, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200 and resp.content:
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
