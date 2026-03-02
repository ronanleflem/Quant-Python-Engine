"""Asset-universe adapters for DCA multi-universe mutualization.

The adapter layer centralizes universe-specific rules (calendar, lot sizing,
fees, corporate actions handling) so the DCA core runner can stay generic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol


UNIVERSE_RULES_VERSION = "asset-universe-rules-v1"


@dataclass(frozen=True)
class UniverseRules:
    calendar: str
    lot_mode: str
    fees_bps: float
    corporate_actions: str
    metadata: Dict[str, Any]


class AssetUniverseAdapter(Protocol):
    """Contract for transforming a generic spec into universe-specific defaults."""

    universe_type: str
    rules_version: str

    def get_rules(self, *, data_spec: Mapping[str, Any], instrument: Mapping[str, Any]) -> UniverseRules:
        ...

    def adapt_data_spec(self, data_spec: Mapping[str, Any], instrument: Mapping[str, Any]) -> Dict[str, Any]:
        ...

    def adapt_context(self, context: Mapping[str, Any], rules: UniverseRules) -> Dict[str, Any]:
        ...


class _BaseUniverseAdapter:
    universe_type = "GENERIC"
    rules_version = UNIVERSE_RULES_VERSION
    default_calendar = "weekday_business_days"
    default_lot_mode = "fractional"
    default_fees_bps = 10.0
    default_corporate_actions = "none"

    def get_rules(self, *, data_spec: Mapping[str, Any], instrument: Mapping[str, Any]) -> UniverseRules:
        cfg = self._extract_universe_config(data_spec, instrument)
        metadata = {
            "timezone": cfg.get("timezone", data_spec.get("timezone", "UTC")),
            "trading_hours": cfg.get("trading_hours", "regular"),
        }
        return UniverseRules(
            calendar=str(cfg.get("calendar", self.default_calendar)),
            lot_mode=str(cfg.get("lot_mode", self.default_lot_mode)),
            fees_bps=float(cfg.get("fees_bps", self.default_fees_bps)),
            corporate_actions=str(cfg.get("corporate_actions", self.default_corporate_actions)),
            metadata=metadata,
        )

    def adapt_data_spec(self, data_spec: Mapping[str, Any], instrument: Mapping[str, Any]) -> Dict[str, Any]:
        adapted = dict(data_spec)
        rules = self.get_rules(data_spec=data_spec, instrument=instrument)
        adapted.setdefault("calendar", rules.calendar)
        adapted.setdefault("lot_mode", rules.lot_mode)
        adapted.setdefault("fees_bps", rules.fees_bps)
        adapted.setdefault("corporate_actions", rules.corporate_actions)
        return adapted

    def adapt_context(self, context: Mapping[str, Any], rules: UniverseRules) -> Dict[str, Any]:
        adapted = dict(context)
        adapted["universe_rules"] = {
            "version": self.rules_version,
            "calendar": rules.calendar,
            "lot_mode": rules.lot_mode,
            "fees_bps": rules.fees_bps,
            "corporate_actions": rules.corporate_actions,
            **rules.metadata,
        }
        return adapted

    @staticmethod
    def _extract_universe_config(data_spec: Mapping[str, Any], instrument: Mapping[str, Any]) -> Mapping[str, Any]:
        raw = instrument.get("universe")
        if isinstance(raw, Mapping):
            return raw
        raw = data_spec.get("universe")
        if isinstance(raw, Mapping):
            return raw
        return {}


class EtfUniverseAdapter(_BaseUniverseAdapter):
    universe_type = "ETF"
    default_calendar = "xetra_business_days"
    default_lot_mode = "integer"
    default_fees_bps = 7.0
    default_corporate_actions = "adjusted_dividends_splits"


class EquityUniverseAdapter(_BaseUniverseAdapter):
    universe_type = "EQUITY"
    default_calendar = "nyse_business_days"
    default_lot_mode = "integer"
    default_fees_bps = 10.0
    default_corporate_actions = "adjusted_splits_dividends"


class CryptoUniverseAdapter(_BaseUniverseAdapter):
    universe_type = "CRYPTO"
    default_calendar = "24x7"
    default_lot_mode = "fractional"
    default_fees_bps = 15.0
    default_corporate_actions = "none"


_REGISTRY: Dict[str, AssetUniverseAdapter] = {
    "ETF": EtfUniverseAdapter(),
    "EQUITY": EquityUniverseAdapter(),
    "ACTION": EquityUniverseAdapter(),
    "CRYPTO": CryptoUniverseAdapter(),
}


def resolve_asset_universe_adapter(asset_class: Optional[str]) -> AssetUniverseAdapter:
    key = str(asset_class or "").strip().upper() or "EQUITY"
    return _REGISTRY.get(key, EquityUniverseAdapter())

