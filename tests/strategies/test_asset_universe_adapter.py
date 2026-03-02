from __future__ import annotations

from quant_engine.datafeeds.asset_universe_adapter import (
    UNIVERSE_RULES_VERSION,
    resolve_asset_universe_adapter,
)


def test_adapter_resolution_and_defaults() -> None:
    etf = resolve_asset_universe_adapter("ETF")
    equity = resolve_asset_universe_adapter("EQUITY")
    crypto = resolve_asset_universe_adapter("CRYPTO")

    etf_rules = etf.get_rules(data_spec={}, instrument={})
    equity_rules = equity.get_rules(data_spec={}, instrument={})
    crypto_rules = crypto.get_rules(data_spec={}, instrument={})

    assert etf_rules.calendar == "xetra_business_days"
    assert equity_rules.calendar == "nyse_business_days"
    assert crypto_rules.calendar == "24x7"
    assert crypto_rules.lot_mode == "fractional"


def test_adapter_overrides_from_spec_and_context() -> None:
    adapter = resolve_asset_universe_adapter("CRYPTO")
    instrument = {
        "universe": {
            "calendar": "24x7",
            "lot_mode": "fractional",
            "fees_bps": 5.0,
            "corporate_actions": "none",
            "timezone": "UTC",
        }
    }
    adapted_data = adapter.adapt_data_spec({"timeframe": "1D"}, instrument)
    rules = adapter.get_rules(data_spec={}, instrument=instrument)
    adapted_ctx = adapter.adapt_context({"symbol": "BTC"}, rules)

    assert adapted_data["fees_bps"] == 5.0
    assert adapted_ctx["universe_rules"]["version"] == UNIVERSE_RULES_VERSION
    assert adapted_ctx["universe_rules"]["calendar"] == "24x7"
