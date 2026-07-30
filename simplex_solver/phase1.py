"""Two-phase method, Phase I: find an initial feasible basis.

Builds the auxiliary problem "maximize -sum(artificial variables)" with the
artificial (or, where available, slack) variables as the starting basis,
and runs it through the same anti-cycling pivoting engine used for Phase
II. If the optimal auxiliary objective is not ~0, the original problem is
infeasible (something the legacy teaching implementation, which assumes
b >= 0 and a trivial slack-only feasible start, has no way to detect or
even represent).
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from simplex_solver.simplex_core import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOL,
    TraceStep,
    _update_basis_inverse,
    run_simplex,
)
from simplex_solver.standardize import StandardForm


@dataclass
class Phase1Result:
    feasible: bool
    basis_indices: List[int]
    non_basis_indices: List[int]
    x_B: np.ndarray
    iterations: int
    trace: Optional[List[TraceStep]] = None


def solve_phase1(
    std: StandardForm,
    *,
    tol: float = DEFAULT_TOL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    collect_trace: bool = False,
) -> Phase1Result:
    m = std.m
    artificial_cols = set(std.artificial_col_for_row.values())

    basis_indices = [0] * m
    for i in range(m):
        if i in std.artificial_col_for_row:
            basis_indices[i] = std.artificial_col_for_row[i]
        elif i in std.slack_col_for_row:
            basis_indices[i] = std.slack_col_for_row[i]
        else:
            raise ValueError(f"Row {i} has neither a slack/surplus nor an artificial column")
    non_basis_indices = sorted(set(range(std.n)) - set(basis_indices))

    if not artificial_cols:
        # Every row started with a legitimate slack in the basis (all <=,
        # rhs >= 0) -- trivially feasible, same starting point as the
        # legacy teaching implementation. No Phase I pivoting needed.
        x_B = np.linalg.inv(std.A[:, basis_indices]) @ std.b
        trace = None
        if collect_trace:
            trace = [TraceStep(
                iteration=0,
                basis_indices=list(basis_indices),
                non_basis_indices=list(non_basis_indices),
                x_B=x_B,
                reduced_costs=np.zeros(len(non_basis_indices)),
                entering_col=None,
                leaving_col=None,
                used_bland=False,
                status="optimal",
            )]
        return Phase1Result(
            feasible=True,
            basis_indices=basis_indices,
            non_basis_indices=non_basis_indices,
            x_B=x_B,
            iterations=0,
            trace=trace,
        )

    phase1_c = np.zeros(std.n)
    for col in artificial_cols:
        phase1_c[col] = -1.0

    result = run_simplex(
        std.A, std.b, phase1_c, basis_indices, non_basis_indices,
        tol=tol, max_iterations=max_iterations, collect_trace=collect_trace,
    )
    assert result.status == "optimal", (
        "Phase I auxiliary objective is bounded above by 0 (sum of non-negative "
        "artificial variables, negated) and should never be reported unbounded"
    )

    phase1_objective = phase1_c[result.basis_indices] @ result.x_B
    if phase1_objective < -tol:
        return Phase1Result(
            feasible=False,
            basis_indices=result.basis_indices,
            non_basis_indices=result.non_basis_indices,
            x_B=result.x_B,
            iterations=result.iterations,
            trace=result.trace,
        )

    basis_indices, non_basis_indices = _drive_out_basic_artificials(
        std.A, result.basis_indices, result.non_basis_indices, artificial_cols, tol
    )
    x_B = np.linalg.inv(std.A[:, basis_indices]) @ std.b

    return Phase1Result(
        feasible=True,
        basis_indices=basis_indices,
        non_basis_indices=non_basis_indices,
        x_B=x_B,
        iterations=result.iterations,
        trace=result.trace,
    )


def _drive_out_basic_artificials(A, basis_indices, non_basis_indices, artificial_cols, tol):
    """Pivot any artificial variables still in the basis (necessarily at
    value ~0 once Phase I is feasible) out in favor of a real column, so
    Phase II's basis contains no artificials. If a row's artificial has no
    real column with a nonzero coefficient, the constraint is redundant and
    the artificial is left in place (harmless: it stays pinned at 0 since
    Phase II blocks artificials from ever re-entering)."""
    basis_indices = list(basis_indices)
    non_basis_indices = list(non_basis_indices)
    A_B_inv = np.linalg.inv(A[:, basis_indices])

    for i in range(len(basis_indices)):
        if basis_indices[i] not in artificial_cols:
            continue
        pivot_col_local = None
        for j_local, col in enumerate(non_basis_indices):
            if col in artificial_cols:
                continue
            if abs((A_B_inv @ A[:, col])[i]) > tol:
                pivot_col_local = j_local
                break
        if pivot_col_local is None:
            continue

        entering_col = non_basis_indices[pivot_col_local]
        d = A_B_inv @ A[:, entering_col]
        A_B_inv = _update_basis_inverse(A_B_inv, d, i)
        basis_indices[i], non_basis_indices[pivot_col_local] = entering_col, basis_indices[i]

    return basis_indices, non_basis_indices
