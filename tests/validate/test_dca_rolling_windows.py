from datetime import datetime, timedelta, timezone

from quant_engine.validate.splitter import generate_dca_rolling_windows


def _dataset(start: datetime, days: int, tz: timezone | None = None) -> list[dict]:
    out = []
    current = start
    for _ in range(days):
        ts = current.replace(tzinfo=tz) if tz else current
        out.append({"timestamp": ts.isoformat()})
        current = current + timedelta(days=1)
    return out


def test_rolling_windows_overlap_and_boundaries() -> None:
    data = _dataset(datetime(2020, 1, 1), days=365 * 7)
    windows = generate_dca_rolling_windows(data, window_years=[3], step_months=12)

    assert windows
    assert windows[0]["start"].startswith("2020-01-01")
    assert windows[0]["end"].startswith("2023-01-01")
    assert windows[1]["start"].startswith("2021-01-01")
    assert windows[1]["end"].startswith("2024-01-01")


def test_rolling_windows_timezone_neutral() -> None:
    start = datetime(2020, 1, 1, 0, 30)
    tz = timezone(timedelta(hours=2))
    data_tz = _dataset(start, days=365 * 4, tz=tz)
    data_naive = _dataset(start, days=365 * 4, tz=None)

    windows_tz = generate_dca_rolling_windows(data_tz, window_years=[3], step_months=12)
    windows_naive = generate_dca_rolling_windows(data_naive, window_years=[3], step_months=12)

    assert [w["start"] for w in windows_tz] == [w["start"] for w in windows_naive]
    assert [w["end"] for w in windows_tz] == [w["end"] for w in windows_naive]


def test_invalid_step_months_raises() -> None:
    data = _dataset(datetime(2020, 1, 1), days=20)
    try:
        generate_dca_rolling_windows(data, window_years=[3], step_months=0)
        assert False, "expected ValueError"
    except ValueError:
        assert True
