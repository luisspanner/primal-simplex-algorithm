"""Sensitivity analysis at an optimal basis: shadow prices and the ranges
over which a constraint's RHS or a variable's objective coefficient can
move without changing which basis is optimal.

All of this falls out of quantities the simplex engine already computes
at the optimal iteration -- result.A_B_inv, result.basis_indices, the
reduced costs -- nothing here re-solves anything.

Limitation: for a `var_mapping.kind == "split"` free variable (one with
no lower or upper bound), cost ranging is reported per standard-form
column (pos_index/neg_index separately) rather than as one range for the
original variable, since a free variable doesn't have a single
unambiguous "cost coefficient" in standard form -- it's split into two
columns whose costs are negatives of each other. This is a documented
gap, not a silently wrong number.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from simplex_solver.problem import LPProblem
from simplex_solver.simplex_core import DEFAULT_TOL, PivotResult
from simplex_solver.standardize import StandardForm


@dataclass
class SensitivityReport:
    shadow_prices: Dict[int, float]  # original constraint index -> y_i (d objective / d b_i)
    rhs_ranges: Dict[int, Tuple[float, float]]  # original constraint index -> (lb, ub) for b_i
    cost_ranges: Dict[int, Tuple[Optional[float], Optional[float]]]  # standard-form column -> (lb, ub) for c_j


def analyze_sensitivity(
    problem: LPProblem,
    std: StandardForm,
    result: PivotResult,
    *,
    tol: float = DEFAULT_TOL,
) -> SensitivityReport:
    """Requires result.status == "optimal" (an optimal basis is the only
    thing sensitivity ranges are defined relative to)."""
    assert result.status == "optimal", (
        "sensitivity analysis requires an optimal PivotResult"
    )

    A_B_inv = result.A_B_inv
    basis_indices = result.basis_indices
    non_basis_indices = result.non_basis_indices
    c_B = std.c[basis_indices]
    c_N = std.c[non_basis_indices]
    A_N = std.A[:, non_basis_indices]
    x_B = result.x_B
    reduced_costs = c_N - c_B @ A_B_inv @ A_N

    y = c_B @ A_B_inv  # shadow price per standard-form row

    shadow_prices: Dict[int, float] = {}
    rhs_ranges: Dict[int, Tuple[float, float]] = {}
    for row in range(std.m):
        constraint_idx = std.row_constraint_index[row]
        if constraint_idx is None:
            continue  # a variable-upper-bound row, not one of the caller's constraints
        sign = std.row_sign[row]
        shadow_prices[constraint_idx] = sign * y[row]
        lb, ub = _rhs_range(A_B_inv, x_B, row, tol)
        # Translate the delta-on-the-(possibly-flipped)-row range back to
        # a delta on the original b_i: flipping the row negates the delta too.
        if sign < 0:
            lb, ub = -ub, -lb
        original_rhs = _original_rhs(std, row)
        rhs_ranges[constraint_idx] = (original_rhs + lb, original_rhs + ub)

    cost_ranges: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for j_local, col in enumerate(non_basis_indices):
        rc = reduced_costs[j_local]
        cost_ranges[col] = (None, std.c[col] - rc)  # c_j can rise to here before it would want to enter

    A_B_inv_A_N = A_B_inv @ A_N
    for i_local, col in enumerate(basis_indices):
        row_coeffs = A_B_inv_A_N[i_local, :]
        lower_deltas = []
        upper_deltas = []
        for j_local, coeff in enumerate(row_coeffs):
            if abs(coeff) <= tol:
                continue
            # Maintain reduced_costs[j] - delta * coeff <= tol (dual feasibility):
            # coeff > 0 => delta >= bound (lower); coeff < 0 => delta <= bound (upper, sign flips on divide).
            bound = reduced_costs[j_local] / coeff
            if coeff > 0:
                lower_deltas.append(bound)
            else:
                upper_deltas.append(bound)
        delta_lb = max(lower_deltas) if lower_deltas else None
        delta_ub = min(upper_deltas) if upper_deltas else None
        lb = None if delta_lb is None else std.c[col] + delta_lb
        ub = None if delta_ub is None else std.c[col] + delta_ub
        cost_ranges[col] = (lb, ub)

    return SensitivityReport(
        shadow_prices=shadow_prices,
        rhs_ranges=rhs_ranges,
        cost_ranges=cost_ranges,
    )


def _rhs_range(A_B_inv: np.ndarray, x_B: np.ndarray, row: int, tol: float) -> Tuple[float, float]:
    """Range of delta (added to b[row]) that keeps x_B + delta * A_B_inv[:, row] >= 0."""
    column = A_B_inv[:, row]
    lower_deltas = []
    upper_deltas = []
    for k, coeff in enumerate(column):
        if abs(coeff) <= tol:
            continue
        bound = -x_B[k] / coeff
        if coeff > 0:
            lower_deltas.append(bound)
        else:
            upper_deltas.append(bound)
    delta_lb = max(lower_deltas) if lower_deltas else -np.inf
    delta_ub = min(upper_deltas) if upper_deltas else np.inf
    return delta_lb, delta_ub


def _original_rhs(std: StandardForm, row: int) -> float:
    """Undo the row flip to recover the original (possibly negative) b_i
    the caller's constraint actually specified."""
    return std.b[row] * std.row_sign[row]
