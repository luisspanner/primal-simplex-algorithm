import json
import os

import pytest

from simplex_solver import LPProblem, solve
from simplex_solver.trace_export import export_trace_json

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name):
    return LPProblem.load_json(os.path.join(FIXTURES_DIR, name))


def test_raises_without_collect_trace():
    problem = _load_fixture("vl.json")
    result = solve(problem)  # collect_trace defaults to False
    with pytest.raises(ValueError, match="collect_trace"):
        export_trace_json(problem, result)


def test_export_round_trips_through_json_for_optimal_fixture():
    problem = _load_fixture("mixed_ge_eq.json")
    result = solve(problem, collect_trace=True)

    exported = export_trace_json(problem, result)
    # must be plain-JSON-serializable (no numpy scalars/arrays left over)
    reloaded = json.loads(json.dumps(exported))

    assert reloaded["status"] == "optimal"
    assert reloaded["x"] == pytest.approx(result.x)
    assert reloaded["objective_value"] == pytest.approx(result.objective_value)
    assert reloaded["iterations"] == result.iterations
    assert len(reloaded["trace"]) == len(result.trace)

    # the embedded problem section must itself be a valid LPProblem
    reconstructed = LPProblem.from_dict(reloaded["problem"])
    assert reconstructed.c == problem.c
    assert reconstructed.sense == problem.sense

    final_step = reloaded["trace"][-1]
    assert final_step["status"] == "optimal"
    assert final_step["x"] == pytest.approx(result.x)


def test_export_handles_infeasible_result_with_null_x():
    problem = _load_fixture("infeasible.json")
    result = solve(problem, collect_trace=True)

    exported = export_trace_json(problem, result)
    reloaded = json.loads(json.dumps(exported))

    assert reloaded["status"] == "infeasible"
    assert reloaded["x"] is None
    assert reloaded["objective_value"] is None
    assert len(reloaded["trace"]) >= 1
