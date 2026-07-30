"""Export a solved LPProblem + its iteration trace as one flat,
self-contained JSON-serializable dict, for a visualization UI to consume.

Requires solve(problem, collect_trace=True) so result.trace is populated.
"""
from typing import Optional

from simplex_solver.problem import LPProblem
from simplex_solver.solver import SimplexResult


def export_trace_json(problem: LPProblem, result: SimplexResult) -> dict:
    if result.trace is None:
        raise ValueError(
            "result.trace is None -- call solve(problem, collect_trace=True) first"
        )

    return {
        "problem": problem.to_dict(),
        "status": result.status.value,
        "x": _floats(result.x),
        "objective_value": _float_or_none(result.objective_value),
        "iterations": result.iterations,
        "trace": [_trace_step_dict(step) for step in result.trace],
    }


def _trace_step_dict(step) -> dict:
    return {
        "phase": step.phase,
        "iteration": step.iteration,
        "x": _floats(step.x),
        "objective_value": float(step.objective_value),
        "entering_col": step.entering_col,
        "leaving_col": step.leaving_col,
        "basis_indices": list(step.basis_indices),
        "used_bland": step.used_bland,
        "status": step.status,
    }


def _floats(values) -> Optional[list]:
    if values is None:
        return None
    return [float(v) for v in values]


def _float_or_none(value) -> Optional[float]:
    return None if value is None else float(value)
