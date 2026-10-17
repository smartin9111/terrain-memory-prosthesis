"""Architecture rules (docs/architecture.md): who may import whom. A violation means the perception/controller work of the two
team members is no longer independent, so this test is the guard of the split."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "terrain_memory_prosthesis"
NAME = "terrain_memory_prosthesis"

# package -> internal packages it may import (itself always allowed)
ALLOWED = {
    "common": set(),
    "perception": {"common"},
    "controllers": {"common"},
    "environment": {"common", "perception"},
    "evaluation": {"common", "environment"},
    "training": {"common", "perception", "controllers", "environment", "evaluation"},
}


def _internal_imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: forbidden style, reported as its own violation
                yield path, "<relative>"
                continue
            mods = [node.module or ""]
            # `from terrain_memory_prosthesis import perception` -> also check the names
            if node.module == NAME:
                mods = [f"{NAME}.{a.name}" for a in node.names]
        for m in mods:
            parts = m.split(".")
            if parts[0] == NAME and len(parts) > 1:
                yield path, parts[1]


def test_import_boundaries():
    bad = []
    for pkg, allowed in ALLOWED.items():
        for f in (PKG / pkg).rglob("*.py"):
            for path, target in _internal_imports(f):
                if target == "<relative>" or (target != pkg and target not in allowed):
                    bad.append(f"{path.relative_to(PKG)} imports {target!r} (allowed for {pkg}: {sorted(allowed | {pkg})})")
    assert not bad, "\n".join(bad)


def test_every_package_is_covered():
    pkgs = {p.name for p in PKG.iterdir() if p.is_dir() and not p.name.startswith("__")}
    assert pkgs == set(ALLOWED), f"new package {pkgs ^ set(ALLOWED)}: add its import rule to ALLOWED"
