import json
import os

import pytest

from simplex_solver.problem import Constraint, LPProblem

TMP_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def make_problem():
    return LPProblem(
        c=[3.0, 2.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[4.0, 2.0], op="<=", rhs=9.0),
            Constraint(coeffs=[10.0, 20.0], op="<=", rhs=51.0),
        ],
    )


def test_round_trip_json(tmp_path):
    problem = make_problem()
    problem.validate()
    path = tmp_path / "problem.json"
    problem.dump_json(str(path))

    loaded = LPProblem.load_json(str(path))
    assert loaded.c == problem.c
    assert loaded.sense == problem.sense
    assert len(loaded.constraints) == len(problem.constraints)
    for a, b in zip(loaded.constraints, problem.constraints):
        assert a.coeffs == b.coeffs
        assert a.op == b.op
        assert a.rhs == b.rhs


def test_round_trip_with_bounds(tmp_path):
    problem = make_problem()
    problem.bounds = [(0.0, None), (None, None)]
    problem.validate()
    path = tmp_path / "problem_bounds.json"
    problem.dump_json(str(path))

    loaded = LPProblem.load_json(str(path))
    assert loaded.bounds == [(0.0, None), (None, None)]


def test_default_bounds_when_omitted():
    problem = make_problem()
    assert problem.bounds is None
    assert problem.effective_bounds() == [(0.0, None), (0.0, None)]


def test_invalid_sense_rejected():
    problem = make_problem()
    problem.sense = "minimizee"
    with pytest.raises(ValueError, match="Invalid sense"):
        problem.validate()


def test_invalid_op_rejected():
    data = {
        "sense": "maximize",
        "c": [1.0, 1.0],
        "constraints": [{"coeffs": [1.0, 1.0], "op": "<", "rhs": 4.0}],
    }
    with pytest.raises(ValueError, match="invalid operator"):
        LPProblem.from_dict(data)


def test_mismatched_row_length_rejected():
    data = {
        "sense": "maximize",
        "c": [1.0, 1.0],
        "constraints": [{"coeffs": [1.0, 1.0, 1.0], "op": "<=", "rhs": 4.0}],
    }
    with pytest.raises(ValueError, match="expected 2 coefficients"):
        LPProblem.from_dict(data)


def test_mismatched_bounds_length_rejected():
    problem = make_problem()
    problem.bounds = [(0.0, None)]
    with pytest.raises(ValueError, match="Expected 2 variable bounds"):
        problem.validate()


def test_inverted_bounds_rejected():
    problem = make_problem()
    problem.bounds = [(5.0, 1.0), (0.0, None)]
    with pytest.raises(ValueError, match="lower bound 5.0 exceeds upper bound 1.0"):
        problem.validate()


def test_missing_required_fields_rejected():
    with pytest.raises(ValueError, match="must contain"):
        LPProblem.from_dict({"c": [1.0]})
