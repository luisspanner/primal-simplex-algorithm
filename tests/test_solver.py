import pytest

from simplex_solver import Constraint, LPProblem, Status, solve


def test_simple_le_problem_matches_legacy_vl_result():
    # Same problem as vl.txt: maximize 3x + 2y
    #   4x + 2y <= 9
    #   10x + 20y <= 51
    #   4x + 3y <= 10
    problem = LPProblem(
        c=[3.0, 2.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[4.0, 2.0], op="<=", rhs=9.0),
            Constraint(coeffs=[10.0, 20.0], op="<=", rhs=51.0),
            Constraint(coeffs=[4.0, 3.0], op="<=", rhs=10.0),
        ],
    )
    result = solve(problem)
    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(7.25)


def test_unbounded_problem_reports_status():
    # maximize x s.t. -x <= 0 (x can grow without bound)
    problem = LPProblem(
        c=[1.0],
        sense="maximize",
        constraints=[Constraint(coeffs=[-1.0], op="<=", rhs=0.0)],
    )
    result = solve(problem)
    assert result.status == Status.UNBOUNDED
    assert result.x is None
    assert result.objective_value is None


def test_infeasible_problem_reports_status():
    problem = LPProblem(
        c=[1.0, 1.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[1.0, 1.0], op="<=", rhs=1.0),
            Constraint(coeffs=[1.0, 1.0], op=">=", rhs=5.0),
        ],
    )
    result = solve(problem)
    assert result.status == Status.INFEASIBLE
    assert result.x is None
    assert result.objective_value is None


def test_mixed_constraints_and_minimize():
    # minimize x + y s.t. x + y >= 4, x - y = 0  -> optimal at x=y=2, obj=4
    problem = LPProblem(
        c=[1.0, 1.0],
        sense="minimize",
        constraints=[
            Constraint(coeffs=[1.0, 1.0], op=">=", rhs=4.0),
            Constraint(coeffs=[1.0, -1.0], op="=", rhs=0.0),
        ],
    )
    result = solve(problem)
    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(4.0)
    assert result.x[0] == pytest.approx(2.0)
    assert result.x[1] == pytest.approx(2.0)
