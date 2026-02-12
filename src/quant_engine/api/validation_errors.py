"""Validation error normalization for HTTP APIs."""
from __future__ import annotations

from typing import Any, Iterable

from pydantic import ValidationError


class ApiValidationException(Exception):
    """Domain-level validation error exposed as HTTP 422."""

    def __init__(self, errors: list[dict[str, str]]):
        super().__init__("validation_error")
        self.errors = errors


def normalize_pydantic_errors(exc: ValidationError) -> list[dict[str, str]]:
    return _normalize_error_items(exc.errors())


def normalize_fastapi_errors(errors: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    return _normalize_error_items(errors)


def single_validation_error(field: str, code: str, message: str) -> list[dict[str, str]]:
    return [{"field": field, "code": code, "message": message}]


def _normalize_error_items(errors: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for err in errors:
        loc = err.get("loc", ())
        if isinstance(loc, tuple):
            loc_values = list(loc)
        elif isinstance(loc, list):
            loc_values = loc
        else:
            loc_values = [loc]
        field = _location_to_field(loc_values)
        code = str(err.get("type") or "validation_error")
        message = str(err.get("msg") or "Invalid value")
        out.append({"field": field, "code": code, "message": message})
    return out


def _location_to_field(loc: list[Any]) -> str:
    if loc and isinstance(loc[0], str) and loc[0] in {"body", "query", "path", "header"}:
        loc = loc[1:]
    if not loc:
        return "request"
    return ".".join(str(part) for part in loc)
