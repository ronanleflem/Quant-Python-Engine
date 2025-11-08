"""Notification helpers for external alerting channels."""

from .telegram_notify import send_telegram_message, is_configured

__all__ = ["send_telegram_message", "is_configured"]
