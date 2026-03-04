from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "quant_engine"

# module -> forbidden top-level package imports
FORBIDDEN_IMPORTS: dict[str, set[str]] = {
    "core": {"api", "backtest", "optimize"},
}


def _module_name_from_path(file_path: Path) -> str:
    relative = file_path.relative_to(SOURCE_ROOT)
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


def _iter_project_import_targets(file_path: Path) -> list[str]:
    module_name = _module_name_from_path(file_path)
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


def test_architecture_forbidden_imports() -> None:
    violations: list[str] = []

    for file_path in SOURCE_ROOT.rglob("*.py"):
        source_module = _module_name_from_path(file_path)
        source_package = source_module.split(".")[1]
        forbidden_targets = FORBIDDEN_IMPORTS.get(source_package, set())
        if not forbidden_targets:
            continue

        for imported_module in _iter_project_import_targets(file_path):
            imported_parts = imported_module.split(".")
            if len(imported_parts) < 2:
                continue
            target_package = imported_parts[1]
            if target_package in forbidden_targets:
                violations.append(
                    f"{source_module} -> {imported_module} interdit "
                    f"(règle: {source_package} ne doit pas dépendre de {target_package})"
                )

    assert not violations, "Dépendances interdites détectées:\n- " + "\n- ".join(sorted(violations))
