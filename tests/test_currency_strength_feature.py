import pandas as pd

from quant_engine.core.features.currency_strength import (
    build_strength_table,
    enrich_with_currency_strength,
    parse_fx_symbol,
)


def _pair(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC"),
            "close": closes,
            "symbol": symbol,
        }
    )


def test_parse_fx_symbol_formats():
    assert parse_fx_symbol("EURUSD") == ("EUR", "USD")
    assert parse_fx_symbol("EUR/USD") == ("EUR", "USD")
    assert parse_fx_symbol("eur_usd") == ("EUR", "USD")


def test_enrich_with_currency_strength_spread_columns_present():
    prices = {
        "EURUSD": _pair("EURUSD", [1.00, 1.01, 1.02, 1.03, 1.04, 1.05]),
        "GBPUSD": _pair("GBPUSD", [1.20, 1.205, 1.21, 1.215, 1.22, 1.225]),
        "EURGBP": _pair("EURGBP", [0.83, 0.831, 0.832, 0.833, 0.834, 0.835]),
    }
    table = build_strength_table(prices, lookback=3)
    assert "ccy_strength_EUR" in table.columns
    assert "ccy_strength_USD" in table.columns

    df = _pair("EURUSD", [1.00, 1.01, 1.02, 1.03, 1.04, 1.05])
    out = enrich_with_currency_strength(df, symbol="EURUSD", prices_by_symbol=prices, lookback=3)
    assert {"ccy_strength_base", "ccy_strength_quote", "ccy_strength_spread"}.issubset(out.columns)
    assert out["ccy_strength_spread"].notna().any()
