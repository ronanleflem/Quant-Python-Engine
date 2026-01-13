"""Shared HTTP client configuration with retries/backoff and pooling."""
from __future__ import annotations

import os
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_CONNECT_TIMEOUT = float(os.getenv("QE_HTTP_CONNECT_TIMEOUT", "3.0"))
DEFAULT_READ_TIMEOUT = float(os.getenv("QE_HTTP_READ_TIMEOUT", "10.0"))
DEFAULT_TIMEOUT = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)
DEFAULT_RETRIES = int(os.getenv("QE_HTTP_RETRIES", "3"))
DEFAULT_BACKOFF = float(os.getenv("QE_HTTP_BACKOFF", "0.4"))
DEFAULT_POOL_CONNECTIONS = int(os.getenv("QE_HTTP_POOL_CONNECTIONS", "10"))
DEFAULT_POOL_MAXSIZE = int(os.getenv("QE_HTTP_POOL_MAXSIZE", "10"))

_SESSION: Optional[requests.Session] = None


def _build_retry() -> Retry:
    return Retry(
        total=DEFAULT_RETRIES,
        connect=DEFAULT_RETRIES,
        read=DEFAULT_RETRIES,
        status=DEFAULT_RETRIES,
        backoff_factor=DEFAULT_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"),
        raise_on_status=False,
    )


def get_shared_session() -> requests.Session:
    """Return a pooled session configured with retries and backoff."""

    global _SESSION
    if _SESSION is None:
        session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=_build_retry(),
            pool_connections=DEFAULT_POOL_CONNECTIONS,
            pool_maxsize=DEFAULT_POOL_MAXSIZE,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _SESSION = session
    return _SESSION


def request_json(
    method: str,
    url: str,
    *,
    timeout: Optional[float | tuple[float, float]] = None,
    session: Optional[requests.Session] = None,
    **kwargs: Any,
) -> Any:
    """Issue an HTTP request and return the parsed JSON payload."""

    http = session or get_shared_session()
    response = http.request(method, url, timeout=timeout or DEFAULT_TIMEOUT, **kwargs)
    response.raise_for_status()
    if not response.content:
        return None
    return response.json()

