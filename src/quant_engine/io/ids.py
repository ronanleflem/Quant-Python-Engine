"""Run identifier utilities."""
from __future__ import annotations

import uuid


def generate_id() -> str:
    return uuid.uuid4().hex

