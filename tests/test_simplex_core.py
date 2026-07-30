import numpy as np
import pytest

from simplex_solver.simplex_core import SimplexDidNotConverge, run_simplex


def _beale_cycling_example():
    """Beale's classic cycling example (see e.g. Chvatal, Linear
    Programming): maximize 0.75x1 - 150x2 + 0.02x3 - 6x4 subject to
        0.25x1 - 60x2 - 0.04x3 + 9x4 <= 0
        0.5x1  - 90x2 - 0.02x3 + 3x4 <= 0
        x3 <= 1
    Under Dantzig's rule (most positive reduced cost, ties broken by lowest
    index), this cycles forever instead of converging. Optimal objective is
    0.05 (verified independently via scipy.optimize.linprog). Variables 0-3
    are the decision vars, 4-6 are the slacks.
    """
    A = np.array([
        [0.25, -60.0, -0.04, 9.0, 1.0, 0.0, 0.0],
        [0.5, -90.0, -0.02, 3.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ])
    b = np.array([0.0, 0.0, 1.0])
    c = np.array([0.75, -150.0, 0.02, -6.0, 0.0, 0.0, 0.0])
    basis_indices = [4, 5, 6]
    non_basis_indices = [0, 1, 2, 3]
    return A, b, c, basis_indices, non_basis_indices


def test_dantzig_rule_alone_cycles_on_beale_example():
    A, b, c, basis_indices, non_basis_indices = _beale_cycling_example()
    with pytest.raises(SimplexDidNotConverge):
        run_simplex(
            A, b, c, basis_indices, non_basis_indices,
            bland_after=10_000,  # effectively never switch to Bland's rule
            max_iterations=30,
        )


def test_blands_rule_resolves_cycling_on_beale_example():
    A, b, c, basis_indices, non_basis_indices = _beale_cycling_example()
    result = run_simplex(
        A, b, c, basis_indices, non_basis_indices,
        bland_after=0,  # use Bland's rule from the first iteration
        max_iterations=100,
    )
    assert result.status == "optimal"
    objective = c[result.basis_indices] @ result.x_B
    assert objective == pytest.approx(0.05)


def test_simple_problem_converges_and_matches_known_optimum():
    # maximize x + 4y s.t. x+y<=4, x<=2, y<=3 (matches vl.txt-style fixture)
    A = np.array([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
    ])
    b = np.array([4.0, 2.0, 3.0])
    c = np.array([1.0, 4.0, 0.0, 0.0, 0.0])
    basis_indices = [2, 3, 4]
    non_basis_indices = [0, 1]

    result = run_simplex(A, b, c, basis_indices, non_basis_indices)
    assert result.status == "optimal"
    objective = c[result.basis_indices] @ result.x_B
    assert objective == pytest.approx(13.0)


def test_unbounded_problem_is_detected():
    # maximize x s.t. -x <= 0 (x can grow without bound)
    A = np.array([[-1.0, 1.0]])
    b = np.array([0.0])
    c = np.array([1.0, 0.0])
    basis_indices = [1]
    non_basis_indices = [0]

    result = run_simplex(A, b, c, basis_indices, non_basis_indices)
    assert result.status == "unbounded"
