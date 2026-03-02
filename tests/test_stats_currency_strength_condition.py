import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "quant_engine" / "stats" / "conditions.py"
_spec = importlib.util.spec_from_file_location("qe_stats_conditions", MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_mod)


currency_strength_regime = _mod.currency_strength_regime


def test_currency_strength_regime_labels():
    df = pd.DataFrame({"ccy_strength_spread": [-0.5, -0.1, 0.0, 0.3]})
    out = currency_strength_regime(df, long_threshold=0.2, short_threshold=-0.2)
    assert list(out.astype(str)) == ["short", "neutral", "neutral", "long"]
