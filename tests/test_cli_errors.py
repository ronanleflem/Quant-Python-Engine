import json
from pathlib import Path
from urllib import error

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from quant_engine.cli import main as cli_main

SPEC_PATH = Path(__file__).parent / "data" / "spec_example.json"


def test_submit_connection_error(monkeypatch) -> None:
    def raise_urlopen(*_args, **_kwargs):
        raise error.URLError("Service down")

    monkeypatch.setattr(cli_main.request, "urlopen", raise_urlopen)
    runner = CliRunner()

    result = runner.invoke(cli_main.app, ["submit", "--spec", str(SPEC_PATH)])

    assert result.exit_code == 1
    assert "Connection error: Service down" in result.stdout


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
