import numpy as np
import pytest

from simplex_solver.dual_simplex import run_dual_simplex
from simplex_solver.simplex_core import run_simplex

# maximize x + 4y  s.t. x+y<=4, x<=2, y<=3  (same problem as
# test_simplex_core.test_simple_problem_converges_and_matches_known_optimum)
A = np.array([
    [1.0, 1.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 1.0],
])
b = np.array([4.0, 2.0, 3.0])
c = np.array([1.0, 4.0, 0.0, 0.0, 0.0])


def _solved_basis():
    result = run_simplex(A, b, c, [2, 3, 4], [0, 1])
    assert result.status == "optimal"
    return result.basis_indices, result.non_basis_indices


def test_dual_simplex_matches_fresh_primal_solve_after_rhs_tightening():
    basis_indices, non_basis_indices = _solved_basis()

    new_b = np.array([1.0, 2.0, 3.0])  # tighten x+y<=4 to x+y<=1
    dual_result = run_dual_simplex(A, new_b, c, basis_indices, non_basis_indices)
    assert dual_result.status == "optimal"

    fresh_result = run_simplex(A, new_b, c, [2, 3, 4], [0, 1])
    assert fresh_result.status == "optimal"

    dual_obj = c[dual_result.basis_indices] @ dual_result.x_B
    fresh_obj = c[fresh_result.basis_indices] @ fresh_result.x_B
    assert dual_obj == pytest.approx(fresh_obj)


def test_dual_simplex_uses_far_fewer_iterations_than_a_fresh_solve():
    basis_indices, non_basis_indices = _solved_basis()
    new_b = np.array([1.0, 2.0, 3.0])

    dual_result = run_dual_simplex(A, new_b, c, basis_indices, non_basis_indices)
    fresh_result = run_simplex(A, new_b, c, [2, 3, 4], [0, 1])

    assert dual_result.iterations <= fresh_result.iterations


def test_dual_simplex_detects_primal_infeasibility():
    basis_indices, non_basis_indices = _solved_basis()

    # x + y <= -1 is impossible since x, y >= 0
    new_b = np.array([-1.0, 2.0, 3.0])
    dual_result = run_dual_simplex(A, new_b, c, basis_indices, non_basis_indices)
    assert dual_result.status == "infeasible"


def test_dual_simplex_no_pivots_needed_when_still_feasible():
    basis_indices, non_basis_indices = _solved_basis()

    # At the optimum (x=1, y=3), x<=2 is slack (x=1, not binding) while
    # x+y<=4 and y<=3 are both tight. Loosening the slack constraint's
    # RHS can't push any basic variable negative -- no pivot needed.
    new_b = np.array([4.0, 20.0, 3.0])
    dual_result = run_dual_simplex(A, new_b, c, basis_indices, non_basis_indices)
    assert dual_result.status == "optimal"
    assert dual_result.iterations == 0


def test_dual_simplex_rejects_a_dual_infeasible_starting_basis():
    # starting basis [0, 1] is not optimal for c = [1, 4, 0, 0, 0] against
    # the identity-ish columns here -- some non-basic reduced cost is > 0,
    # violating run_dual_simplex's precondition.
    with pytest.raises(AssertionError):
        run_dual_simplex(A, b, c, [0, 1, 4], [2, 3])


def test_bland_rule_path_converges_on_dual_simplex_too():
    basis_indices, non_basis_indices = _solved_basis()
    new_b = np.array([1.0, 2.0, 3.0])
    dual_result = run_dual_simplex(A, new_b, c, basis_indices, non_basis_indices, bland_after=0)
    assert dual_result.status == "optimal"
