"""Shared pytest configuration for local test shims.

This project still runs with Starlette's legacy ``TestClient`` implementation,
which passes an ``app=...`` keyword to ``httpx.Client``. ``httpx>=0.28``
removed that argument, so we provide a tiny compatibility shim at test import
time to keep API tests collectable until dependencies are upgraded in lockstep.
"""

import inspect
import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def _install_testclient_httpx_compat() -> None:
    import httpx

    if "app" in inspect.signature(httpx.Client.__init__).parameters:
        return

    original_init = httpx.Client.__init__

    def _compat_init(self, *args, app=None, **kwargs):
        return original_init(self, *args, **kwargs)

    httpx.Client.__init__ = _compat_init


def _install_optional_pymysql_stub() -> None:
    if "pymysql" in sys.modules:
        return
    try:
        __import__("pymysql")
    except ModuleNotFoundError:
        stub = types.ModuleType("pymysql")

        def _missing(*_args, **_kwargs):
            raise ModuleNotFoundError("pymysql is required for MySQL-backed flows")

        cursors = types.ModuleType("pymysql.cursors")

        class DictCursor:  # pragma: no cover - used only for import compatibility
            pass

        cursors.DictCursor = DictCursor
        stub.connect = _missing
        stub.cursors = cursors
        sys.modules["pymysql"] = stub
        sys.modules["pymysql.cursors"] = cursors


_install_testclient_httpx_compat()
_install_optional_pymysql_stub()
