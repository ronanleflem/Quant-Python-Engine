from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "quant_engine"

# module critique -> imports top-level interdits
FORBIDDEN_IMPORTS: dict[str, set[str]] = {
    "core": {"api", "backtest", "cli", "live", "optimize", "performance", "seasonality"},
    "market_intelligence": {"api", "backtest", "cli", "live", "optimize", "performance"},
    "strategies": {"api", "cli"},
    "stats": {"cli"},
}

REMEDIATION_MESSAGE = (
    "Remédiation: déplacer les contrats/types partagés dans quant_engine.core "
    "(ou quant_engine.market_intelligence pour les features de marché), "
    "puis injecter les dépendances depuis les couches d'orchestration (api/cli/live)."
)


def _module_name_from_path(file_path: Path, *, source_root: Path = SOURCE_ROOT) -> str:
    relative = file_path.relative_to(source_root)
    return ".".join(("quant_engine",) + relative.with_suffix("").parts)


def _resolve_imported_module(current_module: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    current_parts = current_module.split(".")
    base_parts = current_parts[:-1]
    if node.level > len(base_parts):
        return None

    resolved_parts = base_parts[: len(base_parts) - node.level + 1]
    if node.module:
        resolved_parts.extend(node.module.split("."))
    return ".".join(resolved_parts)


def _iter_project_import_targets(file_path: Path, *, source_root: Path = SOURCE_ROOT) -> list[str]:
    module_name = _module_name_from_path(file_path, source_root=source_root)
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    targets: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_imported_module(module_name, node)
            if resolved:
                targets.append(resolved)

    return [target for target in targets if target.startswith("quant_engine.")]


def _collect_forbidden_import_violations(
    *,
    source_root: Path,
    forbidden_imports: dict[str, set[str]],
) -> list[str]:
    violations: list[str] = []

    for file_path in source_root.rglob("*.py"):
        source_module = _module_name_from_path(file_path, source_root=source_root)
        source_package = source_module.split(".")[1]
        forbidden_targets = forbidden_imports.get(source_package, set())
        if not forbidden_targets:
            continue

        for imported_module in _iter_project_import_targets(file_path, source_root=source_root):
            imported_parts = imported_module.split(".")
            if len(imported_parts) < 2:
                continue
            target_package = imported_parts[1]
            if target_package in forbidden_targets:
                violations.append(
                    f"{source_module} -> {imported_module} interdit "
                    f"(règle: {source_package} ne doit pas dépendre de {target_package})"
                )

    return sorted(violations)


def test_architecture_forbidden_imports() -> None:
    violations = _collect_forbidden_import_violations(
        source_root=SOURCE_ROOT,
        forbidden_imports=FORBIDDEN_IMPORTS,
    )

    assert not violations, (
        "Dépendances interdites détectées:\n- "
        + "\n- ".join(violations)
        + "\n\n"
        + REMEDIATION_MESSAGE
    )


def test_architecture_forbidden_imports_detects_synthetic_violation(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_engine"
    core_dir = source_root / "core"
    api_dir = source_root / "api"
    core_dir.mkdir(parents=True)
    api_dir.mkdir(parents=True)

    (source_root / "__init__.py").write_text("", encoding="utf-8")
    (core_dir / "__init__.py").write_text("", encoding="utf-8")
    (api_dir / "__init__.py").write_text("", encoding="utf-8")
    (core_dir / "service.py").write_text("from quant_engine.api import app\n", encoding="utf-8")

    violations = _collect_forbidden_import_violations(
        source_root=source_root,
        forbidden_imports={"core": {"api"}},
    )

    assert violations == [
        "quant_engine.core.service -> quant_engine.api interdit "
        "(règle: core ne doit pas dépendre de api)"
    ]


def test_architecture_forbidden_imports_remediation_message() -> None:
    message = "Dépendances interdites détectées:\n- foo\n\n" + REMEDIATION_MESSAGE

    assert "Remédiation:" in message
    assert "injecter les dépendances" in message
