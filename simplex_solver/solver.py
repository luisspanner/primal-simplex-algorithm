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
from simplex_solver.simplex_core import TraceStep, run_simplex
from simplex_solver.standardize import StandardForm, standardize


class Status(Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"


@dataclass
class SolverTraceStep:
    """One TraceStep, translated into the caller's original variable space
    (via StandardForm.recover_original_x/recover_objective) so viewers
    never need to know about slack/surplus/artificial columns."""
    phase: int  # 1 or 2
    iteration: int
    x: List[float]
    objective_value: float
    entering_col: Optional[int]
    leaving_col: Optional[int]
    basis_indices: List[int]
    used_bland: bool
    status: Optional[str]


@dataclass
class SimplexResult:
    status: Status
    x: Optional[List[float]]
    objective_value: Optional[float]
    iterations: int
    trace: Optional[List[SolverTraceStep]] = None


def _to_solver_trace_step(std: StandardForm, step: TraceStep, phase: int) -> SolverTraceStep:
    y_full = np.zeros(std.n)
    for idx, val in zip(step.basis_indices, step.x_B):
        y_full[idx] = val
    return SolverTraceStep(
        phase=phase,
        iteration=step.iteration,
        x=std.recover_original_x(y_full),
        objective_value=std.recover_objective(std.c @ y_full),
        entering_col=step.entering_col,
        leaving_col=step.leaving_col,
        basis_indices=list(step.basis_indices),
        used_bland=step.used_bland,
        status=step.status,
    )


def solve(problem: LPProblem, *, collect_trace: bool = False) -> SimplexResult:
    problem.validate()
    std = standardize(problem)

    phase1_result = solve_phase1(std, collect_trace=collect_trace)
    trace: Optional[List[SolverTraceStep]] = [] if collect_trace else None
    if trace is not None and phase1_result.trace is not None:
        trace.extend(_to_solver_trace_step(std, step, phase=1) for step in phase1_result.trace)

    if not phase1_result.feasible:
        return SimplexResult(
            status=Status.INFEASIBLE, x=None, objective_value=None,
            iterations=phase1_result.iterations, trace=trace,
        )

    artificial_cols = frozenset(std.artificial_col_for_row.values())
    phase2_result = run_simplex(
        std.A, std.b, std.c,
        phase1_result.basis_indices, phase1_result.non_basis_indices,
        disallowed_entering=artificial_cols, collect_trace=collect_trace,
    )
    if trace is not None and phase2_result.trace is not None:
        trace.extend(_to_solver_trace_step(std, step, phase=2) for step in phase2_result.trace)

    total_iterations = phase1_result.iterations + phase2_result.iterations

    if phase2_result.status == "unbounded":
        return SimplexResult(
            status=Status.UNBOUNDED, x=None, objective_value=None,
            iterations=total_iterations, trace=trace,
        )

    y_full = np.zeros(std.n)
    for idx, val in zip(phase2_result.basis_indices, phase2_result.x_B):
        y_full[idx] = val

    x_original = std.recover_original_x(y_full)
    objective_value = std.recover_objective(std.c @ y_full)

    return SimplexResult(
        status=Status.OPTIMAL, x=x_original, objective_value=objective_value,
        iterations=total_iterations, trace=trace,
    )
