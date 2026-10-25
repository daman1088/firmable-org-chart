"""Ground-truth isolation guard.

Inference-side code (normalisation, inference, the Streamlit explorer) must only see
observed data. This test statically scans those sources and fails if they import
the generator, the truth storage module, evaluation or model-fitting code, or if
they reference the truth directory, truth tables / attributes or truth columns.

The single exception is app/evaluation_view.py, the dashboard's explicitly
separated evaluation mode. The explorer may import it only lazily, inside the
evaluation branch, and that is checked below too.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INFERENCE_SIDE = [REPO / "src/orgchart/normalization", REPO / "src/orgchart/inference", REPO / "app"]
EVALUATION_MODE = REPO / "app" / "evaluation_view.py"
FORBIDDEN_MODULES = ("orgchart.generation", "orgchart.storage.truth", "orgchart.evaluation", "orgchart.learning")
FORBIDDEN_STRINGS = ("truth/", "/truth", "truth_", "manager_id", "nearest_observed_manager", "canonical_title",
                     "true_manager", "true_ancestor")


def _python_files() -> list[Path]:
    return [p for root in INFERENCE_SIDE if root.exists() for p in root.rglob("*.py") if p != EVALUATION_MODE]


def _imports(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            names += [f"{node.module}.{a.name}" for a in node.names]
    return names


def _violations(source: str) -> list[str]:
    tree = ast.parse(source)
    problems = [f"import {m}" for m in _imports(tree) if m.startswith(FORBIDDEN_MODULES)]
    for node in ast.walk(tree):
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        if name and "truth" in name.lower():
            problems.append(f"name {name!r}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            problems += [f"string mentions {s!r}" for s in FORBIDDEN_STRINGS if s in node.value.lower()
                         and not (isinstance(node, ast.Constant) and _is_docstring(tree, node))]
    return problems


def _is_docstring(tree: ast.AST, const: ast.Constant) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and body[0].value is const:
                return True
    return False


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_inference_side_never_touches_truth(path):
    assert not _violations(path.read_text()), f"{path}: {_violations(path.read_text())}"


def test_guard_detects_violations():
    assert _violations("from orgchart.storage.truth import connect_with_truth")
    assert _violations("import orgchart.generation.catalog")
    assert _violations("from orgchart.storage import truth")
    assert _violations("x = 'data/full/truth/employees.parquet'")
    assert _violations("cols = ['manager_id']")
    assert _violations("p = paths.truth_employees")
    assert _violations("con.sql('SELECT * FROM truth_employees')")
    assert _violations("from orgchart.learning.fit import fit_models")
    assert not _violations('"""Docs may say truth."""\nfrom orgchart.storage.observed import connect_observed')
    assert not _violations("label = 'Evaluation (ground truth)'")


def test_evaluation_mode_is_imported_lazily():
    tree = ast.parse((REPO / "app" / "streamlit_app.py").read_text())
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top_level for a in n.names] + [n.module for n in top_level if isinstance(n, ast.ImportFrom)]
    assert "evaluation_view" not in names
    nested = [n for n in ast.walk(tree) if isinstance(n, ast.Import) and any(a.name == "evaluation_view" for a in n.names)]
    assert nested, "evaluation mode should be imported inside the evaluation branch"


def test_observed_storage_module_has_no_truth_access():
    source = (REPO / "src/orgchart/storage/observed.py").read_text()
    assert "truth" not in source.split('"""', 2)[-1].lower()
