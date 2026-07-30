import os

import pytest

from simplex_solver.phase1 import solve_phase1
from simplex_solver.problem import Constraint, LPProblem
from simplex_solver.sensitivity import analyze_sensitivity
from simplex_solver.simplex_core import run_simplex
from simplex_solver.standardize import standardize

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _solve(problem):
    std = standardize(problem)
    phase1_result = solve_phase1(std)
    assert phase1_result.feasible
    artificial_cols = frozenset(std.artificial_col_for_row.values())
    result = run_simplex(
        std.A, std.b, std.c,
        phase1_result.basis_indices, phase1_result.non_basis_indices,
        disallowed_entering=artificial_cols,
    )
    assert result.status == "optimal"
    return std, result


def _scipy_reference(problem):
    """Solve via scipy and return (shadow prices per original constraint
    index, sign-adjusted to this project's max-problem convention)."""
    linprog = pytest.importorskip("scipy.optimize").linprog

    n = problem.n_vars
    A_ub, b_ub, ub_rows = [], [], []
    A_eq, b_eq, eq_rows = [], [], []
    for i, constraint in enumerate(problem.constraints):
        if constraint.op == "<=":
            A_ub.append(constraint.coeffs)
            b_ub.append(constraint.rhs)
            ub_rows.append(i)
        elif constraint.op == ">=":
            A_ub.append([-x for x in constraint.coeffs])
            b_ub.append(-constraint.rhs)
            ub_rows.append(i)
        elif constraint.op == "=":
            A_eq.append(constraint.coeffs)
            b_eq.append(constraint.rhs)
            eq_rows.append(i)

    c = problem.c if problem.sense == "minimize" else [-x for x in problem.c]
    res = linprog(
        c, A_ub=A_ub or None, b_ub=b_ub or None,
        A_eq=A_eq or None, b_eq=b_eq or None,
        bounds=[(0, None)] * n, method="highs",
    )
    if res.status != 0:
        return None, None

    # scipy's marginals are d(scipy's minimized objective)/d(rhs); this
    # project reports d(original maximize/minimize objective)/d(rhs), so
    # negate when we negated c above (i.e. whenever sense == "maximize").
    sign = -1.0 if problem.sense == "maximize" else 1.0
    shadow_prices = {}
    for local_i, row in enumerate(ub_rows):
        shadow_prices[row] = sign * res.ineqlin.marginals[local_i]
    for local_i, row in enumerate(eq_rows):
        shadow_prices[row] = sign * res.eqlin.marginals[local_i]
    return res.x, shadow_prices


@pytest.mark.parametrize(
    "fixture_name",
    ["basic_lp.json", "vl.json", "mixed_ge_eq.json"],
)
def test_shadow_prices_match_scipy_marginals(fixture_name):
    problem = LPProblem.load_json(os.path.join(FIXTURES_DIR, fixture_name))
    std, result = _solve(problem)
    report = analyze_sensitivity(problem, std, result)

    _, expected_shadow_prices = _scipy_reference(problem)
    if expected_shadow_prices is None:
        pytest.skip("scipy unavailable or reported non-optimal status")

    for row, expected in expected_shadow_prices.items():
        assert report.shadow_prices[row] == pytest.approx(expected, abs=1e-6)


def test_hand_checked_rhs_and_cost_ranges():
    # maximize 2x1 + 3x2 s.t. x1+x2<=4, x1+2x2<=5, x1,x2>=0
    # optimal: x1=3, x2=1, obj=9, basis=[x1, x2]
    # A_B_inv = [[2,-1],[-1,1]] (hand-derived from A_B=[[1,1],[1,2]])
    # RHS ranging (hand-verified via x_B(delta) = x_B + delta*A_B_inv[:,row] >= 0):
    #   row0 (b=4): delta in [-1.5, 1]   -> b1 in [2.5, 5.0]
    #   row1 (b=5): delta in [-1, 3]     -> b2 in [4.0, 8.0]
    # cost ranging (hand-verified via reduced-cost-preserving delta):
    #   c1 (=2): delta in [-0.5, 1] -> [1.5, 3.0]
    #   c2 (=3): delta in [-1, 1]   -> [2.0, 4.0]
    problem = LPProblem(
        c=[2.0, 3.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[1.0, 1.0], op="<=", rhs=4.0),
            Constraint(coeffs=[1.0, 2.0], op="<=", rhs=5.0),
        ],
    )
    std, result = _solve(problem)
    report = analyze_sensitivity(problem, std, result)

    assert report.shadow_prices[0] == pytest.approx(1.0)
    assert report.shadow_prices[1] == pytest.approx(1.0)

    assert report.rhs_ranges[0] == pytest.approx((2.5, 5.0))
    assert report.rhs_ranges[1] == pytest.approx((4.0, 8.0))

    # basis columns are [0, 1] (x1, x2) in standard form
    assert report.cost_ranges[0] == pytest.approx((1.5, 3.0))
    assert report.cost_ranges[1] == pytest.approx((2.0, 4.0))


def test_analyze_sensitivity_requires_optimal_result():
    problem = LPProblem.load_json(os.path.join(FIXTURES_DIR, "infeasible.json"))
    std = standardize(problem)
    phase1_result = solve_phase1(std)
    assert not phase1_result.feasible

    from simplex_solver.simplex_core import PivotResult
    fake_infeasible_result = PivotResult(
        status="infeasible",
        basis_indices=phase1_result.basis_indices,
        non_basis_indices=phase1_result.non_basis_indices,
        x_B=phase1_result.x_B,
        iterations=phase1_result.iterations,
        A_B_inv=None,
    )
    with pytest.raises(AssertionError):
        analyze_sensitivity(problem, std, fake_infeasible_result)
