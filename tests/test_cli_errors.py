import json
from pathlib import Path

import requests

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from quant_engine.cli import main as cli_main

SPEC_PATH = Path(__file__).parent / "data" / "spec_example.json"


def test_submit_connection_error(monkeypatch) -> None:
    def raise_request(*_args, **_kwargs):
        raise requests.ConnectionError("Service down")

    monkeypatch.setattr(cli_main, "request_json", raise_request)
    runner = CliRunner()

    result = runner.invoke(cli_main.app, ["submit", "--spec", str(SPEC_PATH)])

    assert result.exit_code == 1
    assert "Connection error: Service down" in result.stdout


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (400, "Bad Request"),
        (500, "Server Error"),
    ],
)
def test_submit_http_error(monkeypatch, status, reason) -> None:
    def raise_request(*_args, **_kwargs):
        raise requests.HTTPError(f"HTTP {status}: {reason}")

    monkeypatch.setattr(cli_main, "request_json", raise_request)
    runner = CliRunner()

    result = runner.invoke(cli_main.app, ["submit", "--spec", str(SPEC_PATH)])

    assert result.exit_code == 1
    assert f"HTTP {status}: {reason}" in result.stdout


def test_run_local_outputs_json() -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app,
        ["run-local", "--spec", str(SPEC_PATH)],
        env={"DB_DSN": "sqlite:///:memory:"},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)


def test_run_local_outputs_minimal_payload(monkeypatch) -> None:
    class DummyJobManager:
        @staticmethod
        def submit(_spec, synchronous=True):
            return {
                "best": {"metrics": {"sharpe": 1.2}, "params": {"foo": 1}, "extra": 2},
                "ignored": True,
            }

    monkeypatch.setattr(cli_main, "JobManager", DummyJobManager)
    runner = CliRunner()

    result = runner.invoke(cli_main.app, ["run-local", "--spec", str(SPEC_PATH)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"metrics": {"sharpe": 1.2}, "params": {"foo": 1}}


def test_run_local_missing_spec() -> None:
    runner = CliRunner()

    result = runner.invoke(cli_main.app, ["run-local", "--spec", "missing.json"])

    assert result.exit_code != 0
    assert "Invalid value for '--spec'" in result.output
