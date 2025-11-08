"""Telegram notification helper."""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = os.getenv("ENABLE_TELEGRAM_ALERTS", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def is_configured() -> bool:
    """Return ``True`` if Telegram notifications are fully configured."""

    return TELEGRAM_ENABLED and bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_CHAT_ID)


def send_telegram_message(text: str, *, parse_mode: Optional[str] = "Markdown") -> None:
    """Send ``text`` to the configured Telegram chat.

    Environment variables required:
    - ``TELEGRAM_BOT_TOKEN`` : Token du bot Telegram.
    - ``TELEGRAM_CHAT_ID`` : Identifiant du chat (utilisateur ou groupe).
    - ``ENABLE_TELEGRAM_ALERTS`` : (optionnel) ``true`` par défaut, mettre à ``false`` pour désactiver.

    Les erreurs sont loggées sans interrompre le flux principal.
    """

    if not TELEGRAM_ENABLED:
        logger.debug("[Telegram] Alerts disabled via ENABLE_TELEGRAM_ALERTS")
        return

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning(
            "[Telegram] Configuration incomplète (TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquant)."
        )
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        resp = requests.post(url, json=payload, timeout=3)
        if resp.status_code != 200:
            logger.error("[Telegram] Échec envoi message (%s): %s", resp.status_code, resp.text)
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("[Telegram] Exception lors de l'envoi du message: %s", exc)

