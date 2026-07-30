"""Anti-cycling, incrementally-updated primal simplex pivoting engine.

Same reduced-cost / ratio-test / column-swap pivoting math as the legacy
teaching implementation (simplex_solver.legacy_teaching.simplex_step), but:
  - uses an epsilon tolerance instead of exact <= 0 / > 0 comparisons, since
    Phase I can hand this degenerate bases with near-zero values
  - falls back to Bland's rule (smallest index, both entering and leaving)
    after a configurable number of iterations, guaranteeing termination on
    degenerate/cycling problems
  - raises a clear error instead of looping forever if it still fails to
    converge within a hard iteration cap
  - maintains the basis inverse via the revised-simplex product-form-of-
    the-inverse (PFI) update (O(m^2) per iteration) instead of recomputing
    np.linalg.inv(A_B) from scratch every iteration (O(m^3)), with periodic
    refactorization to bound floating-point drift. legacy_teaching.py keeps
    the from-scratch approach on purpose, as a naive baseline to compare
    against.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

DEFAULT_TOL = 1e-9
DEFAULT_MAX_ITERATIONS = 10_000
DEFAULT_REFACTORIZE_EVERY = 100


class SimplexDidNotConverge(Exception):
    pass


@dataclass
class PivotResult:
    status: str  # "optimal" or "unbounded"
    basis_indices: List[int]
    non_basis_indices: List[int]
    x_B: np.ndarray
    iterations: int


def run_simplex(
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
) -> PivotResult:
    """Run primal simplex pivots to optimality (or detect unboundedness),
    starting from the given feasible basis. Does not mutate its inputs."""
    m, n = A.shape
    basis_indices = list(basis_indices)
    non_basis_indices = list(non_basis_indices)
    if bland_after is None:
        bland_after = max(50, 10 * n)

    A_B_inv = np.linalg.inv(A[:, basis_indices])

    for iteration in range(max_iterations):
        A_N = A[:, non_basis_indices]
        c_B = c[basis_indices]
        c_N = c[non_basis_indices]
        x_B = A_B_inv @ b
        reduced_costs = c_N - c_B @ A_B_inv @ A_N

        use_bland = iteration >= bland_after
        entering_local = _choose_entering(reduced_costs, non_basis_indices, tol, use_bland)
        if entering_local is None:
            return PivotResult(
                status="optimal",
                basis_indices=basis_indices,
                non_basis_indices=non_basis_indices,
                x_B=x_B,
                iterations=iteration,
            )

        d = A_B_inv @ A_N[:, entering_local]
        if np.all(d <= tol):
            return PivotResult(
                status="unbounded",
                basis_indices=basis_indices,
                non_basis_indices=non_basis_indices,
                x_B=x_B,
                iterations=iteration,
            )

        leaving_local = _choose_leaving(x_B, d, basis_indices, tol, use_bland)

        A_B_inv = _update_basis_inverse(A_B_inv, d, leaving_local)

        basis_indices[leaving_local], non_basis_indices[entering_local] = (
            non_basis_indices[entering_local],
            basis_indices[leaving_local],
        )

        if (iteration + 1) % refactorize_every == 0:
            A_B_inv = np.linalg.inv(A[:, basis_indices])

    raise SimplexDidNotConverge(f"Simplex did not converge after {max_iterations} iterations")


def _update_basis_inverse(A_B_inv: np.ndarray, d: np.ndarray, leaving_local: int) -> np.ndarray:
    """Product-form-of-the-inverse update: given the direction vector
    d = A_B_inv @ A_N[:, entering] and the pivot row (leaving_local),
    return the inverse for the basis after swapping in the entering
    column, without recomputing np.linalg.inv from scratch."""
    pivot_row = A_B_inv[leaving_local, :] / d[leaving_local]
    updated = A_B_inv - np.outer(d, pivot_row)
    updated[leaving_local, :] = pivot_row
    return updated


def _choose_entering(reduced_costs, non_basis_indices, tol, use_bland) -> int:
    candidates = [j for j, rc in enumerate(reduced_costs) if rc > tol]
    if not candidates:
        return None
    if use_bland:
        return min(candidates, key=lambda j: non_basis_indices[j])
    return max(candidates, key=lambda j: reduced_costs[j])


def _choose_leaving(x_B, d, basis_indices, tol, use_bland) -> int:
    m = len(x_B)
    ratios = np.full(m, np.inf)
    for i in range(m):
        if d[i] > tol:
            ratios[i] = x_B[i] / d[i]
    min_ratio = ratios.min()
    tied = [i for i in range(m) if ratios[i] <= min_ratio + tol]
    if use_bland:
        return min(tied, key=lambda i: basis_indices[i])
    return tied[0]
