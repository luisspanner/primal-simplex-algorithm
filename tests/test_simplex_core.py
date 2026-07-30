import numpy as np
import pytest

from simplex_solver.simplex_core import SimplexDidNotConverge, _update_basis_inverse, run_simplex


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


def test_pfi_update_matches_full_reinversion():
    rng = np.random.default_rng(42)
    m = 6
    A_B = rng.integers(1, 10, size=(m, m)).astype(float)
    A_B_inv = np.linalg.inv(A_B)

    entering_col = rng.integers(1, 10, size=m).astype(float)
    d = A_B_inv @ entering_col
    leaving_local = 2

    updated_inv = _update_basis_inverse(A_B_inv, d, leaving_local)

    # Independently recompute the inverse of the basis after swapping in
    # entering_col at column `leaving_local`, the slow/naive way.
    A_B_new = A_B.copy()
    A_B_new[:, leaving_local] = entering_col
    expected_inv = np.linalg.inv(A_B_new)

    assert np.allclose(updated_inv, expected_inv)


def test_pfi_update_matches_scipy_on_larger_random_problem():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(7)
    m, n_decision = 20, 20
    A_dec = rng.integers(1, 10, size=(m, n_decision)).astype(float)
    b = rng.integers(50, 200, size=m).astype(float)
    c = rng.integers(1, 20, size=n_decision).astype(float)

    A = np.hstack([A_dec, np.eye(m)])
    c_full = np.hstack([c, np.zeros(m)])
    basis_indices = list(range(n_decision, n_decision + m))
    non_basis_indices = list(range(n_decision))

    result = run_simplex(A, b, c_full, basis_indices, non_basis_indices)
    assert result.status == "optimal"
    objective = c_full[result.basis_indices] @ result.x_B

    scipy_result = scipy_optimize.linprog(
        -c, A_ub=A_dec, b_ub=b, bounds=[(0, None)] * n_decision, method="highs"
    )
    assert scipy_result.status == 0
    assert objective == pytest.approx(-scipy_result.fun, rel=1e-6)


def test_trace_default_off_and_result_unaffected():
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
    assert result.trace is None


def test_trace_final_entry_matches_result():
    A = np.array([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
    ])
    b = np.array([4.0, 2.0, 3.0])
    c = np.array([1.0, 4.0, 0.0, 0.0, 0.0])
    basis_indices = [2, 3, 4]
    non_basis_indices = [0, 1]

    result = run_simplex(A, b, c, basis_indices, non_basis_indices, collect_trace=True)
    assert result.trace is not None
    assert len(result.trace) == result.iterations + 1

    final_step = result.trace[-1]
    assert final_step.status == result.status
    assert final_step.entering_col is None
    assert final_step.leaving_col is None
    assert final_step.basis_indices == result.basis_indices
    assert np.allclose(final_step.x_B, result.x_B)

    # every non-terminal step records an actual pivot
    for step in result.trace[:-1]:
        assert step.status is None
        assert step.entering_col is not None
        assert step.leaving_col is not None


def test_trace_captures_bland_rule_usage_on_beale_example():
    A, b, c, basis_indices, non_basis_indices = _beale_cycling_example()
    result = run_simplex(
        A, b, c, basis_indices, non_basis_indices,
        bland_after=0, max_iterations=100, collect_trace=True,
    )
    assert result.status == "optimal"
    assert all(step.used_bland for step in result.trace)
