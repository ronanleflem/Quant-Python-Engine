from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_MARKET_STATS_PACK_CONDITION: Dict[str, Any] | None = None


MARKET_STATS_PACKS: Dict[str, Dict[str, Any]] = {
    "candle_structure": {
        "name": "candle_structure",
        "label": "Candle Structure",
        "description": "Single-candle and short streak structure patterns against next-candle and candle-shape targets.",
        "events": [
            {"name": "bullish_candle", "params": {}},
            {"name": "bearish_candle", "params": {}},
            {"name": "bullish_engulfing", "params": {}},
            {"name": "bearish_engulfing", "params": {}},
            {"name": "bullish_streak", "params": {"k": 3}},
            {"name": "bearish_streak", "params": {"k": 3}},
        ],
        "targets": [
            {"name": "next_bullish", "params": {}},
            {"name": "next_bearish", "params": {}},
            {"name": "body_ratio", "params": {}},
            {"name": "upper_wick_ratio", "params": {}},
            {"name": "lower_wick_ratio", "params": {}},
        ],
    },
    "volatility_shocks": {
        "name": "volatility_shocks",
        "label": "Volatility Shocks",
        "description": "ATR shock and streak events against continuation and reversal-style targets.",
        "events": [
            {"name": "shock_atr", "params": {"mult": 2.0, "window": 14}},
            {"name": "k_consecutive", "params": {"k": 2, "direction": "up"}},
            {"name": "k_consecutive", "params": {"k": 2, "direction": "down"}},
        ],
        "targets": [
            {"name": "up_next_bar", "params": {}},
            {"name": "continuation_n", "params": {"n": 3, "direction": "up"}},
            {"name": "continuation_n", "params": {"n": 3, "direction": "down"}},
            {"name": "time_to_reversal", "params": {"max_horizon": 5}},
            {"name": "candle_zscore", "params": {"window": 20}},
        ],
    },
    "gaps_breakouts": {
        "name": "gaps_breakouts",
        "label": "Gaps And Breakouts",
        "description": "Gap and breakout events against breakout-first and retracement targets.",
        "events": [
            {"name": "gap_up", "params": {}},
            {"name": "gap_down", "params": {}},
            {"name": "breakout_hhll", "params": {"lookback": 20, "type": "high"}},
            {"name": "breakout_hhll", "params": {"lookback": 20, "type": "low"}},
        ],
        "targets": [
            {"name": "up_next_bar", "params": {}},
            {"name": "breakout_high_first", "params": {"lookback": 20, "horizon": 10}},
            {"name": "breakout_low_first", "params": {"lookback": 20, "horizon": 10}},
            {"name": "retracement_probability", "params": {"lookback": 20, "horizon": 10, "direction": "up"}},
            {"name": "retracement_probability", "params": {"lookback": 20, "horizon": 10, "direction": "down"}},
        ],
    },
}

MARKET_STATS_PACK_ALIASES = {
    "volatility": "volatility_shocks",
    "volatility_pack": "volatility_shocks",
    "candles": "candle_structure",
    "candle_patterns": "candle_structure",
    "gaps": "gaps_breakouts",
    "breakouts": "gaps_breakouts",
}


def _normalized_pack_key(name: str | None) -> str:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return MARKET_STATS_PACK_ALIASES.get(key, key)


def is_supported_market_stats_pack(name: str | None) -> bool:
    key = _normalized_pack_key(name)
    return key in MARKET_STATS_PACKS or key == "all_basic"


def resolve_market_stats_pack(name: str | None) -> Dict[str, Any]:
    key = _normalized_pack_key(name)
    if key == "all_basic":
        events: List[Dict[str, Any]] = []
        targets: List[Dict[str, Any]] = []
        for pack_key in ("candle_structure", "volatility_shocks", "gaps_breakouts"):
            pack = MARKET_STATS_PACKS[pack_key]
            events.extend(pack["events"])
            targets.extend(pack["targets"])
        return {
            "name": "all_basic",
            "label": "All Basic",
            "description": "Union of candle_structure, volatility_shocks, and gaps_breakouts.",
            "events": _dedupe_specs(events),
            "targets": _dedupe_specs(targets),
        }
    if key not in MARKET_STATS_PACKS:
        supported = ", ".join(sorted(list(MARKET_STATS_PACKS.keys()) + ["all_basic"]))
        raise ValueError(f"Unsupported market_stats data.stats_pack '{name}'. Supported packs: {supported}")
    pack = MARKET_STATS_PACKS[key]
    return {
        "name": pack["name"],
        "label": pack["label"],
        "description": pack["description"],
        "events": _dedupe_specs(pack["events"]),
        "targets": _dedupe_specs(pack["targets"]),
    }


def list_market_stats_packs() -> List[Dict[str, Any]]:
    names = sorted(list(MARKET_STATS_PACKS.keys()) + ["all_basic"])
    return [resolve_market_stats_pack(name) for name in names]


def _dedupe_specs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        name = str(item.get("name") or "").strip()
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        signature = (name, repr(sorted(params.items())))
        if not name or signature in seen:
            continue
        seen.add(signature)
        deduped.append({"name": name, "params": dict(params)})
    return deduped
