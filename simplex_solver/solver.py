"""High-level solve() entry point: general LP in, status + solution out.

Orchestrates standardize -> Phase I -> Phase II -> map back to the
caller's original variables, replacing the legacy teaching implementation's
single entry point (which only ever handled max, <=, b>=0, x>=0 and had no
way to report infeasibility or a clean unbounded status).
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from simplex_solver.phase1 import solve_phase1
from simplex_solver.problem import LPProblem
from simplex_solver.simplex_core import run_simplex
from simplex_solver.standardize import standardize


class Status(Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"


@dataclass
class SimplexResult:
    status: Status
    x: Optional[List[float]]
    objective_value: Optional[float]
    iterations: int


def solve(problem: LPProblem) -> SimplexResult:
    problem.validate()
    std = standardize(problem)

    phase1_result = solve_phase1(std)
    if not phase1_result.feasible:
        return SimplexResult(
            status=Status.INFEASIBLE, x=None, objective_value=None,
            iterations=phase1_result.iterations,
        )

    artificial_cols = frozenset(std.artificial_col_for_row.values())
    phase2_result = run_simplex(
        std.A, std.b, std.c,
        phase1_result.basis_indices, phase1_result.non_basis_indices,
        disallowed_entering=artificial_cols,
    )
    total_iterations = phase1_result.iterations + phase2_result.iterations

    if phase2_result.status == "unbounded":
        return SimplexResult(
            status=Status.UNBOUNDED, x=None, objective_value=None,
            iterations=total_iterations,
        )

    y_full = np.zeros(std.n)
    for idx, val in zip(phase2_result.basis_indices, phase2_result.x_B):
        y_full[idx] = val

    x_original = std.recover_original_x(y_full)
    objective_value = std.recover_objective(std.c @ y_full)

    return SimplexResult(
        status=Status.OPTIMAL, x=x_original, objective_value=objective_value,
        iterations=total_iterations,
    )
