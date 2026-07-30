"""Dual simplex: re-optimize from a dual-feasible, primal-infeasible basis.

Mirror image of simplex_core.run_simplex. Primal simplex starts from a
primal-feasible basis (x_B >= 0) and pivots toward dual feasibility
(reduced costs <= 0). Dual simplex starts from a basis that is already
dual-feasible (reduced costs already <= 0, i.e. would be optimal if only
x_B were >= 0) and pivots toward primal feasibility, by kicking out the
most-infeasible basic variable each iteration.

Its use case isn't "an alternate way to solve an LP from scratch" (it
can't -- the all-slack basis for a fresh problem is rarely dual-feasible)
but re-optimizing after a change to `b` that leaves a previously-optimal
basis dual-feasible but primal-infeasible (see solver.resolve_after_rhs_change).

Reuses TraceStep/PivotResult/_update_basis_inverse/SimplexDidNotConverge
from simplex_core -- the pivot-update math is identical regardless of
which rule picked the pivot, only entering/leaving selection differs.
"""
from typing import FrozenSet, List

import numpy as np

from simplex_solver.simplex_core import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_REFACTORIZE_EVERY,
    DEFAULT_TOL,
    PivotResult,
    SimplexDidNotConverge,
    TraceStep,
    _update_basis_inverse,
)


def run_dual_simplex(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    basis_indices: List[int],
    non_basis_indices: List[int],
    *,
    tol: float = DEFAULT_TOL,
    bland_after: int = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    refactorize_every: int = DEFAULT_REFACTORIZE_EVERY,
    disallowed_entering: FrozenSet[int] = frozenset(),
    collect_trace: bool = False,
) -> PivotResult:
    """Run dual simplex pivots to primal feasibility (+ optimality), or
    detect primal infeasibility, starting from a dual-feasible basis.
    Does not mutate its inputs.

    Raises AssertionError if the starting basis is not dual-feasible --
    unlike run_simplex, there's no "just start from the origin" fallback
    here, so a violated precondition is a caller bug, not a case to
    accommodate.

    PivotResult.status is one of "optimal", "infeasible" (the primal
    problem has no feasible solution -- a row demands improvement but no
    non-basic column can supply it) -- there's no "unbounded" status for
    dual simplex; primal unboundedness shows up as dual infeasibility,
    which is exactly the precondition this function assumes away.
    """
    m, n = A.shape
    basis_indices = list(basis_indices)
    non_basis_indices = list(non_basis_indices)
    if bland_after is None:
        bland_after = max(50, 10 * n)

    trace: List[TraceStep] = [] if collect_trace else None
    A_B_inv = np.linalg.inv(A[:, basis_indices])

    for iteration in range(max_iterations):
        A_N = A[:, non_basis_indices]
        c_B = c[basis_indices]
        c_N = c[non_basis_indices]
        x_B = A_B_inv @ b
        reduced_costs = c_N - c_B @ A_B_inv @ A_N

        if iteration == 0:
            assert np.all(reduced_costs <= tol), (
                "run_dual_simplex requires a dual-feasible starting basis "
                "(all reduced costs <= tol); got a positive reduced cost, "
                "which means primal simplex -- not dual -- is the correct "
                "engine for this starting basis"
            )

        use_bland = iteration >= bland_after
        leaving_local = _choose_leaving_dual(x_B, basis_indices, tol, use_bland)

        if leaving_local is None:
            if trace is not None:
                trace.append(TraceStep(
                    iteration=iteration,
                    basis_indices=list(basis_indices),
                    non_basis_indices=list(non_basis_indices),
                    x_B=x_B,
                    reduced_costs=reduced_costs,
                    entering_col=None,
                    leaving_col=None,
                    used_bland=use_bland,
                    status="optimal",
                ))
            return PivotResult(
                status="optimal",
                basis_indices=basis_indices,
                non_basis_indices=non_basis_indices,
                x_B=x_B,
                iterations=iteration,
                A_B_inv=A_B_inv,
                trace=trace,
            )

        alpha_row = A_B_inv[leaving_local, :] @ A_N
        entering_local = _choose_entering_dual(
            reduced_costs, non_basis_indices, alpha_row, tol, use_bland, disallowed_entering
        )

        if entering_local is None:
            if trace is not None:
                trace.append(TraceStep(
                    iteration=iteration,
                    basis_indices=list(basis_indices),
                    non_basis_indices=list(non_basis_indices),
                    x_B=x_B,
                    reduced_costs=reduced_costs,
                    entering_col=None,
                    leaving_col=basis_indices[leaving_local],
                    used_bland=use_bland,
                    status="infeasible",
                ))
            return PivotResult(
                status="infeasible",
                basis_indices=basis_indices,
                non_basis_indices=non_basis_indices,
                x_B=x_B,
                iterations=iteration,
                A_B_inv=A_B_inv,
                trace=trace,
            )

        if trace is not None:
            trace.append(TraceStep(
                iteration=iteration,
                basis_indices=list(basis_indices),
                non_basis_indices=list(non_basis_indices),
                x_B=x_B,
                reduced_costs=reduced_costs,
                entering_col=non_basis_indices[entering_local],
                leaving_col=basis_indices[leaving_local],
                used_bland=use_bland,
            ))

        d = A_B_inv @ A_N[:, entering_local]
        A_B_inv = _update_basis_inverse(A_B_inv, d, leaving_local)

        basis_indices[leaving_local], non_basis_indices[entering_local] = (
            non_basis_indices[entering_local],
            basis_indices[leaving_local],
        )

        if (iteration + 1) % refactorize_every == 0:
            A_B_inv = np.linalg.inv(A[:, basis_indices])

    raise SimplexDidNotConverge(
        f"Dual simplex did not converge after {max_iterations} iterations", trace=trace
    )


def _choose_leaving_dual(x_B, basis_indices, tol, use_bland) -> int:
    """Pick the basic row to fix next: the one most primal-infeasible
    (most negative x_B), or under Bland's rule, the violating row whose
    basic variable has the smallest original column index."""
    violating = [i for i in range(len(x_B)) if x_B[i] < -tol]
    if not violating:
        return None
    if use_bland:
        return min(violating, key=lambda i: basis_indices[i])
    return min(violating, key=lambda i: x_B[i])


def _choose_entering_dual(reduced_costs, non_basis_indices, alpha_row, tol, use_bland, disallowed_entering=frozenset()) -> int:
    """Ratio test on the leaving row: among non-basic columns that would
    pull the leaving row's value back toward feasibility (alpha_row < 0),
    pick the one with the smallest -reduced_cost/alpha_row ratio (keeps
    all other reduced costs <= 0, i.e. preserves dual feasibility)."""
    candidates = [
        j for j, a in enumerate(alpha_row)
        if a < -tol and non_basis_indices[j] not in disallowed_entering
    ]
    if not candidates:
        return None
    if use_bland:
        return min(candidates, key=lambda j: non_basis_indices[j])
    return min(candidates, key=lambda j: -reduced_costs[j] / alpha_row[j])
