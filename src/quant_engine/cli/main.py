"""Command line interface entry points."""
from __future__ import annotations

from pathlib import Path
import typer

from ..core import spec
from ..optimize import runner

app = typer.Typer()


@app.command("run-local")
def run_local(spec_path: Path) -> None:
    sp = spec.load_spec(spec_path)
    runner.run(sp)


if __name__ == "__main__":
    app()
