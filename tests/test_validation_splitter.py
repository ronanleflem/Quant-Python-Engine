from datetime import datetime, timedelta, timezone

from quant_engine.validate.splitter import generate_folds


def _build_dataset(start: datetime, days: int, tz: timezone | None = None) -> list[dict]:
    rows = []
    current = start
    for _ in range(days):
        if tz is not None:
            current = current.replace(tzinfo=tz)
        rows.append({"timestamp": current.isoformat()})
        current = (current + timedelta(days=1)).replace(tzinfo=None)
    return rows


def test_empty_dataset_returns_empty_list() -> None:
    assert generate_folds([], train_months=1, test_months=1, folds=2, embargo_days=0) == []


def test_timezone_aware_conversion_is_neutral() -> None:
    start = datetime(2020, 1, 1, 0, 30)
    tz = timezone(timedelta(hours=2))
    dataset_tz = _build_dataset(start, days=40, tz=tz)
    dataset_naive = _build_dataset(start, days=40, tz=None)

    folds_tz = generate_folds(
        dataset_tz, train_months=1, test_months=1, folds=1, embargo_days=0
    )
    folds_naive = generate_folds(
        dataset_naive, train_months=1, test_months=1, folds=1, embargo_days=0
    )

    def _normalise(rows: list[dict]) -> list[datetime]:
        normalised: list[datetime] = []
        for row in rows:
            dt = datetime.fromisoformat(row["timestamp"])
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            normalised.append(dt)
        return normalised

    assert _normalise(folds_tz[0]["train"]) == _normalise(folds_naive[0]["train"])
    assert _normalise(folds_tz[0]["test"]) == _normalise(folds_naive[0]["test"])


def test_embargo_days_is_applied() -> None:
    dataset = _build_dataset(datetime(2020, 1, 1), days=70)
    folds = generate_folds(dataset, train_months=1, test_months=1, folds=1, embargo_days=2)

    test_rows = folds[0]["test"]
    assert test_rows[0]["timestamp"].startswith("2020-02-03")
    assert all(
        not row["timestamp"].startswith(("2020-02-01", "2020-02-02"))
        for row in test_rows
    )


def test_breaks_when_test_rows_empty() -> None:
    dataset = _build_dataset(datetime(2020, 1, 1), days=15)
    folds = generate_folds(dataset, train_months=1, test_months=1, folds=3, embargo_days=0)
    assert folds == []


def test_timezone_aware_utc_offsets_are_supported() -> None:
    dataset = _build_dataset(datetime(2020, 1, 1), days=40, tz=timezone.utc)

    folds = generate_folds(dataset, train_months=1, test_months=1, folds=1, embargo_days=0)

    assert len(folds) == 1
    assert folds[0]["train"]
    assert folds[0]["test"]


def test_train_test_boundaries_with_embargo() -> None:
    dataset = _build_dataset(datetime(2020, 1, 1), days=62)

    folds = generate_folds(dataset, train_months=1, test_months=1, folds=1, embargo_days=2)

    train_rows = folds[0]["train"]
    test_rows = folds[0]["test"]

    assert train_rows[0]["timestamp"].startswith("2020-01-01")
    assert train_rows[-1]["timestamp"].startswith("2020-01-31")
    assert test_rows[0]["timestamp"].startswith("2020-02-03")
    assert test_rows[-1]["timestamp"].startswith("2020-03-02")
