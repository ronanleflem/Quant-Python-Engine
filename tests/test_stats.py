import pandas as pd

from quant_engine.stats import events, conditions, targets, runner, estimators
from quant_engine.api import schemas


def load_df():
    import json, pathlib
    data = json.loads(pathlib.Path('tests/data/ohlcv.json').read_text())
    df = pd.DataFrame(data)
    df.rename(columns={'timestamp': 'ts', 'session': 'session_id'}, inplace=True)
    return df

def test_events():
    df = load_df()
    k = events.k_consecutive(df, k=3, direction='up')
    assert k.tolist() == [False, False, True, True, True]
    shock = events.shock_atr(df, mult=0.5, window=2)
    assert shock.tolist() == [False, True, True, True, True]
    br = events.breakout_hhll(df, lookback=2, type='hh')
    assert br.tolist() == [False, True, True, True, True]


def test_conditions():
    df = load_df()
    trend = conditions.htf_trend(df, tf_multiplier=2, ema_period=2)
    assert trend.tolist() == ['down', 'down', 'up', 'up', 'up']
    vol = conditions.vol_tertile(df, window=2)
    assert vol.tolist() == ['mid']*5
    sess = conditions.session(df, col='session_id')
    assert sess.tolist() == [
        '2020-01-01',
        '2020-01-02',
        '2020-01-03',
        '2020-01-04',
        '2020-01-05',
    ]


def test_targets():
    df = load_df()
    up = targets.up_next_bar(df)
    assert up.tolist() == [True, True, True, True, pd.NA]
    cont = targets.continuation_n(df, n=2, direction='up')
    assert cont.tolist() == [True, True, True, pd.NA, pd.NA]
    ttr = targets.time_to_reversal(df, max_horizon=3)
    assert ttr.tolist() == [3, 3, pd.NA, pd.NA, pd.NA]


def test_freq_with_wilson():
    p, low, high = estimators.freq_with_wilson(50, 100)
    assert round(p, 3) == 0.5
    assert low < p < high


def test_aggregate_binary():
    s = pd.Series([True, False, True, True])
    res = estimators.aggregate_binary(s)
    assert res["n"] == 4
    assert res["successes"] == 3
    assert res["p_hat"] == 0.75
    assert res["ci_low"] < res["p_hat"] < res["ci_high"]


def test_runner_summary():
    spec = schemas.StatsSpec(
        data=schemas.StatsDataSpec(
            dataset_path='tests/data/ohlcv.json',
            symbols=['ABC'],
            timeframe='1d',
            start='2020-01-01',
            end='2020-01-05',
        ),
        events=[schemas.StatsEventSpec(name='k_consecutive', params={'k':2,'direction':'up'})],
        targets=[schemas.StatsTargetSpec(name='up_next_bar')],
    )
    df = runner.run_stats(spec)
    cols = [
        'symbol',
        'event',
        'condition_name',
        'condition_value',
        'target',
        'n',
        'successes',
        'p_hat',
        'ci_low',
        'ci_high',
        'lift',
        'insufficient',
        'split',
    ]
    assert list(df.columns) == cols
    assert len(df) == 1
    row = df.iloc[0]
    assert row['n'] == 3
    assert row['successes'] == 3
    assert row['lift'] == 0.0
    assert row['insufficient'] == True
    assert row['split'] == 'test'
