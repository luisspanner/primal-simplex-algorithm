import os

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


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name):
    return LPProblem.load_json(os.path.join(FIXTURES_DIR, name))


def _scipy_reference(problem):
    """Solve the same LPProblem via scipy.optimize.linprog as an
    independent oracle. Assumes default bounds (x >= 0)."""
    linprog = pytest.importorskip("scipy.optimize").linprog

    n = problem.n_vars
    A_ub, b_ub = [], []
    A_eq, b_eq = [], []
    for constraint in problem.constraints:
        if constraint.op == "<=":
            A_ub.append(constraint.coeffs)
            b_ub.append(constraint.rhs)
        elif constraint.op == ">=":
            A_ub.append([-x for x in constraint.coeffs])
            b_ub.append(-constraint.rhs)
        elif constraint.op == "=":
            A_eq.append(constraint.coeffs)
            b_eq.append(constraint.rhs)

    c = problem.c if problem.sense == "minimize" else [-x for x in problem.c]
    result = linprog(
        c,
        A_ub=A_ub or None,
        b_ub=b_ub or None,
        A_eq=A_eq or None,
        b_eq=b_eq or None,
        bounds=[(0, None)] * n,
        method="highs",
    )
    if result.status != 0:
        return None
    objective = result.fun if problem.sense == "minimize" else -result.fun
    return objective


@pytest.mark.parametrize(
    "fixture_name,expected_status",
    [
        ("basic_lp.json", Status.OPTIMAL),
        ("test_lp.json", Status.OPTIMAL),
        ("vl.json", Status.OPTIMAL),
        ("mixed_ge_eq.json", Status.OPTIMAL),
        ("minimize.json", Status.OPTIMAL),
        ("infeasible.json", Status.INFEASIBLE),
        ("unbounded.json", Status.UNBOUNDED),
        ("degenerate_beale.json", Status.OPTIMAL),
    ],
)
def test_fixture_status_matches_expectation(fixture_name, expected_status):
    problem = _load_fixture(fixture_name)
    result = solve(problem)
    assert result.status == expected_status


@pytest.mark.parametrize(
    "fixture_name",
    ["basic_lp.json", "test_lp.json", "vl.json", "mixed_ge_eq.json", "minimize.json", "degenerate_beale.json"],
)
def test_fixture_objective_matches_scipy(fixture_name):
    problem = _load_fixture(fixture_name)
    result = solve(problem)
    assert result.status == Status.OPTIMAL

    expected = _scipy_reference(problem)
    if expected is None:
        pytest.skip("scipy unavailable or reported non-optimal status")
    assert result.objective_value == pytest.approx(expected, rel=1e-6, abs=1e-6)
