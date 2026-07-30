import numpy as np
import pytest

from simplex_solver.phase1 import solve_phase1
from simplex_solver.problem import Constraint, LPProblem
from simplex_solver.standardize import standardize


def _x_full(std, result):
    x = np.zeros(std.n)
    for idx, val in zip(result.basis_indices, result.x_B):
        x[idx] = val
    return x


def test_all_le_problem_is_trivially_feasible_without_pivoting():
    problem = LPProblem(
        c=[1.0, 4.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[1.0, 1.0], op="<=", rhs=4.0),
            Constraint(coeffs=[1.0, 0.0], op="<=", rhs=2.0),
        ],
    )
    std = standardize(problem)
    result = solve_phase1(std)
    assert result.feasible
    assert result.iterations == 0


def test_feasible_mixed_ge_and_eq_constraints():
    # x, y >= 0
    #   x + y <= 10
    #   x - y  = 2
    #   x + 2y >= 3
    problem = LPProblem(
        c=[1.0, 1.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[1.0, 1.0], op="<=", rhs=10.0),
            Constraint(coeffs=[1.0, -1.0], op="=", rhs=2.0),
            Constraint(coeffs=[1.0, 2.0], op=">=", rhs=3.0),
        ],
    )
    std = standardize(problem)
    result = solve_phase1(std)
    assert result.feasible

    x = _x_full(std, result)
    assert np.allclose(std.A @ x, std.b, atol=1e-8)
    for col in std.artificial_col_for_row.values():
        assert abs(x[col]) < 1e-8

    orig_x = std.recover_original_x(x)
    assert orig_x[0] - orig_x[1] == pytest.approx(2.0)


def test_infeasible_problem_detected():
    # x, y >= 0 with x + y <= 1 and x + y >= 5 simultaneously: impossible.
    problem = LPProblem(
        c=[1.0, 1.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[1.0, 1.0], op="<=", rhs=1.0),
            Constraint(coeffs=[1.0, 1.0], op=">=", rhs=5.0),
        ],
    )
    std = standardize(problem)
    result = solve_phase1(std)
    assert not result.feasible


def test_equality_only_infeasible_system():
    # x + y = 1 and x + y = 5 can never both hold.
    problem = LPProblem(
        c=[1.0, 1.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[1.0, 1.0], op="=", rhs=1.0),
            Constraint(coeffs=[1.0, 1.0], op="=", rhs=5.0),
        ],
    )
    std = standardize(problem)
    result = solve_phase1(std)
    assert not result.feasible
