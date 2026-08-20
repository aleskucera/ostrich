"""Static check that call sites still match the ostrich public API.

Parses -- never imports -- every Python file under examples/, experiments/,
src/ and tests/, and checks two things against the installed package:

  1. every ``from ostrich[.x] import Name`` names something that exists
  2. every ``SomeOstrichConfig(kw=...)`` uses real dataclass field names

Both drifted silently for weeks: ``ExecutionConfig`` was deleted from the
public API and ``OstrichEngineConfig`` went from flat kwargs to nested
sub-configs, while four experiment sweeps kept importing and constructing the
old shapes. Nothing noticed until someone tried to run them.

Static rather than importing the files, because 44 of them have no
``__main__`` guard and would do real work on import, and some need optional
dependencies (genesis, brax) that are not installed. The trade is that this
catches "that name no longer exists", not value or behavior drift -- a config
field whose *meaning* changed still slips through.
"""

import ast
import dataclasses
import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN_DIRS = ("examples", "experiments", "src/ostrich", "tests")


def _python_files():
    for directory in SCAN_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


FILES = list(_python_files())
FILE_IDS = [str(p.relative_to(ROOT)) for p in FILES]

# The two diagnostics below import ostrich.logging.hdf5_reader.HDF5Reader,
# which commit 09664f9 deleted as a "dead wrapper module" while these two
# consumers were still using it (get_dataset / get_attribute / list_attributes).
# Restoring them needs a replacement reader, not a mechanical port, so they are
# listed rather than fixed.
#
# The xfail is strict and narrowed to AssertionError: fixing one of these makes
# the suite fail until it is removed from the list, and a file that fails to
# parse at all still errors instead of hiding in here (which is how committed
# IndentationErrors in helhest_batch/ went unnoticed).
STALE_IMPORTS = frozenset({
    "experiments/2_dt_stability/diagnose.py",
    "experiments/2_dt_stability/diagnose_helhest_drop.py",
})

STALE_KWARGS = frozenset()

STALE_REASON = "uses HDF5Reader, deleted in 09664f9 with no replacement"


def _params(stale: frozenset):
    return [
        pytest.param(path, id=file_id, marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=STALE_REASON))
        if file_id in stale
        else pytest.param(path, id=file_id)
        for path, file_id in zip(FILES, FILE_IDS)
    ]


def _is_ostrich(module: str | None) -> bool:
    return bool(module) and (module == "ostrich" or module.startswith("ostrich."))


def _ostrich_imports(tree: ast.AST):
    """Yield (module, imported_name, lineno) for absolute ostrich imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and _is_ostrich(node.module):
            for alias in node.names:
                if alias.name != "*":
                    yield node.module, alias.name, alias.asname, node.lineno


def _resolves(module: str, name: str) -> bool:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return False
    if hasattr(mod, name):
        return True
    # `from ostrich.core import engine_config` imports a submodule, which is
    # not an attribute of the parent until it has been imported.
    try:
        importlib.import_module(f"{module}.{name}")
    except ImportError:
        return False
    return True


@pytest.mark.parametrize("path", _params(STALE_IMPORTS))
def test_ostrich_imports_resolve(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    missing = [
        f"line {lineno}: from {module} import {name}"
        for module, name, _asname, lineno in _ostrich_imports(tree)
        if not _resolves(module, name)
    ]
    assert not missing, "imports no longer in the ostrich API:\n  " + "\n  ".join(missing)


def _dataclass_fields(module: str, name: str) -> set[str] | None:
    """Field names of an ostrich dataclass, or None if it is not one."""
    try:
        obj = getattr(importlib.import_module(module), name, None)
    except ImportError:
        return None
    if obj is None or not dataclasses.is_dataclass(obj):
        return None
    return {f.name for f in dataclasses.fields(obj)}


@pytest.mark.parametrize("path", _params(STALE_KWARGS))
def test_ostrich_config_kwargs_exist(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))

    # Local name -> ostrich dataclass fields, for names this file imported.
    fields_by_local_name = {}
    for module, name, asname, _lineno in _ostrich_imports(tree):
        fields = _dataclass_fields(module, name)
        if fields is not None:
            fields_by_local_name[asname or name] = (name, fields)

    # A local class or function of the same name shadows the import; skip those
    # rather than report a call to something that is not the ostrich class.
    shadowed = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }

    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        entry = fields_by_local_name.get(node.func.id)
        if entry is None or node.func.id in shadowed:
            continue
        # `**kwargs` hides the actual names; nothing to check.
        if any(kw.arg is None for kw in node.keywords):
            continue
        cls, fields = entry
        for kw in node.keywords:
            if kw.arg not in fields:
                bad.append(f"line {node.lineno}: {cls}({kw.arg}=...)")

    assert not bad, "config kwargs no longer in the dataclass:\n  " + "\n  ".join(bad)
