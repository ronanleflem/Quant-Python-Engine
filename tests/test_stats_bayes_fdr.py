import os
from statistics import NormalDist

from quant_engine.stats import estimators
from quant_engine.persistence import session
from quant_engine.api import app
from quant_engine.config import reset_settings_cache


def test_bayes_simple():
    res = estimators.aggregate_binary_bayes(60, 100)
    assert abs(res["p_mean"] - 0.6) < 0.02
    assert res["hdi_low"] < res["p_mean"] < res["hdi_high"]


def test_fdr_bh():
    if not hasattr(NormalDist, "sf"):
        NormalDist.sf = lambda self, x: 1 - self.cdf(x)
    n = 500
    p0 = 0.5
    successes = [int(0.58 * n)] * 5 + [int(0.5 * n)] * 15
    pvals = [
        estimators.p_value_binomial_onesided_normal(s, n, p0) for s in successes
    ]
    qvals = estimators.benjamini_hochberg(pvals)
    sig = [q <= 0.05 for q in qvals]
    assert sum(sig[:5]) > 5 / 2
    assert sum(sig[5:]) < 15 / 2


def setup_db(tmp_path):
    db_path = tmp_path / "stats.sqlite"
    os.environ["DB_DSN"] = f"sqlite:///{db_path}"
    reset_settings_cache()
    return db_path


def test_api_list_stats_significant_bayes(tmp_path):
    setup_db(tmp_path)
    with session() as conn:
        conn.execute("ALTER TABLE market_stats ADD COLUMN q_value REAL")
        conn.execute("ALTER TABLE market_stats ADD COLUMN lift_bayes REAL")
        conn.execute("ALTER TABLE market_stats ADD COLUMN significant INTEGER")
        rows = [
            (
                "ABC",
                "1h",
                "ev1",
                "cond",
                "A",
                "up",
                "test",
                10,
                6,
                0.6,
                0.5,
                0.7,
                0.2,
                "2020",
                "2020",
                0.04,
                0.2,
                1,
            ),
            (
                "ABC",
                "1h",
                "ev2",
                "cond",
                "B",
                "up",
                "test",
                10,
                7,
                0.7,
                0.6,
                0.8,
                0.3,
                "2020",
                "2020",
                0.02,
                0.3,
                1,
            ),
            (
                "ABC",
                "1h",
                "ev3",
                "cond",
                "C",
                "up",
                "test",
                10,
                5,
                0.5,
                0.4,
                0.6,
                0.1,
                "2020",
                "2020",
                0.2,
                0.4,
                0,
            ),
        ]
        conn.executemany(
            "INSERT INTO market_stats (symbol,timeframe,event,condition_name,condition_value,target,split,n,successes,p_hat,ci_low,ci_high,lift,start,end,q_value,lift_bayes,significant) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    res = app.list_stats(
        symbol="ABC", timeframe="1h", significant_only=True, method="bayes"
    )
    assert [r["event"] for r in res] == ["ev2", "ev1"]
