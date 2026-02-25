from quant_engine.persistence import db


def test_prepare_sql_mysql_rewrites_insert_or_replace() -> None:
    sql = """
    INSERT OR REPLACE INTO api_jobs(job_id, job_type, status)
    VALUES (?, ?, ?)
    """
    prepared = db._prepare_sql_for_mysql(sql)
    assert "INSERT INTO api_jobs" in prepared
    assert "ON DUPLICATE KEY UPDATE" in prepared
    assert "%s" in prepared


def test_prepare_sql_mysql_rewrites_on_conflict_do_update() -> None:
    sql = """
    INSERT INTO market_stats(symbol, timeframe, event)
    VALUES (?, ?, ?)
    ON CONFLICT(symbol, timeframe, event) DO UPDATE SET
        event = excluded.event
    """
    prepared = db._prepare_sql_for_mysql(sql)
    assert "ON DUPLICATE KEY UPDATE" in prepared
    assert "event = VALUES(event)" in prepared
    assert "%s" in prepared


def test_prepare_sql_mysql_rewrites_on_conflict_do_nothing() -> None:
    sql = """
    INSERT INTO trials(run_id, trial_number)
    VALUES (?, ?)
    ON CONFLICT(run_id, trial_number) DO NOTHING
    """
    prepared = db._prepare_sql_for_mysql(sql)
    assert "INSERT IGNORE INTO trials" in prepared
    assert "ON CONFLICT" not in prepared
    assert "%s" in prepared
