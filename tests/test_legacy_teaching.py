import os

from simplex_solver.legacy_teaching import simplex

FIXTURES_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(FIXTURES_DIR)


def test_vl_smoke():
    filename = os.path.join(REPO_ROOT, "vl.txt")
    x_B, x_N, obj_value, optimal, basis_indices, non_basis_indices = simplex(filename)
    assert optimal
    assert obj_value == 7.25
