"""Cross-check legacy_teaching.simplex() against simplex_solver.solve() on
fixtures both can handle (canonical <=, b >= 0, maximize) -- the comparison
harness the legacy module was kept around for."""
import os

import pytest

from simplex_solver import LPProblem, Status, solve
from simplex_solver.legacy_teaching import simplex as legacy_simplex

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
LEGACY_TXT_FILES = {
    "basic_lp.json": "basic_lp.txt",
    "test_lp.json": "test_lp.txt",
    "vl.json": "vl.txt",
}
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.mark.parametrize("json_fixture,txt_fixture", LEGACY_TXT_FILES.items())
def test_legacy_and_new_solver_agree_on_objective(json_fixture, txt_fixture):
    txt_path = os.path.join(REPO_ROOT, txt_fixture)
    if not os.path.exists(txt_path):
        pytest.skip(f"{txt_fixture} not present in repo root")

    _, _, legacy_objective, optimal, _, _ = legacy_simplex(txt_path)
    assert optimal

    problem = LPProblem.load_json(os.path.join(FIXTURES_DIR, json_fixture))
    result = solve(problem)
    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(legacy_objective)
