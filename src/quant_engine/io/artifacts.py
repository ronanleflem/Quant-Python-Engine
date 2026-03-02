"""Utilities to persist backtest and statistics results."""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _json_default(obj: Any) -> Any:
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _write_rows(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(rows, default=_json_default))


def write_trials(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    _write_rows(path, rows)


def write_trades(path: str | Path, trades: List[Dict[str, Any]]) -> None:
    _write_rows(path, trades)


def write_equity(path: str | Path, equity: List[float]) -> None:
    rows = [{"equity": v} for v in equity]
    _write_rows(path, rows)


def write_summary(path: str | Path, summary: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(summary, indent=2, default=_json_default))


def write_stats_summary(path: str | Path, df: "pd.DataFrame") -> None:
    """Persist aggregate statistics to a Parquet file."""

    import pandas as pd

    if not df.empty:
        df = df.copy()
        for col in ("n", "successes"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    df.to_parquet(path, index=False)


def write_stats_details(path: str | Path, df: "pd.DataFrame") -> None:
    """Persist detailed statistics to a Parquet file.

    Placeholder for future extensions (e.g. time to reversal).
    """

    df.to_parquet(path, index=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_hash(dataset_rows: List[Dict[str, Any]]) -> str:
    payload = json.dumps(dataset_rows, sort_keys=True, default=_json_default).encode("utf-8")
    return sha256_bytes(payload)


def compute_spec_hash(spec: Any) -> str:
    if hasattr(spec, "model_dump"):
        spec_payload = spec.model_dump(mode="json")
    elif isinstance(spec, dict):
        spec_payload = spec
    else:
        spec_payload = str(spec)
    encoded = json.dumps(spec_payload, sort_keys=True, default=_json_default).encode("utf-8")
    return sha256_bytes(encoded)


def get_runtime_versions() -> Dict[str, str]:
    import pandas as pd

    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_run_manifest(path: str | Path, manifest: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))


def write_checksums(path: str | Path, files: List[str | Path]) -> None:
    out_lines: List[str] = []
    for file_path in sorted(Path(f) for f in files):
        out_lines.append(f"{sha256_file(file_path)}  {file_path.name}")
    Path(path).write_text("\n".join(out_lines) + "\n")
