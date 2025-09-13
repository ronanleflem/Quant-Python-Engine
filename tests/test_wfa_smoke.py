from pathlib import Path

from quant_engine.core import spec
from quant_engine.optimize import runner


def test_wfa_creates_artifacts(tmp_path: Path):
    sp = spec.load_spec("tests/data/spec_example.json")
    result = runner.run(sp, out_dir=tmp_path)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "trials.parquet").exists()
    assert result["best"] is not None
