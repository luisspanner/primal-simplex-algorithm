import numpy as np
import pytest

from simplex_solver.problem import Constraint, LPProblem
from simplex_solver.standardize import standardize


def test_all_le_matches_legacy_shape():
    problem = LPProblem(
        c=[1.0, 4.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[1.0, 1.0], op="<=", rhs=4.0),
            Constraint(coeffs=[1.0, 0.0], op="<=", rhs=2.0),
            Constraint(coeffs=[0.0, 1.0], op="<=", rhs=3.0),
        ],
    )
    std = standardize(problem)
    assert std.n_decision == 2
    assert std.A.shape == (3, 2 + 3)  # 2 decision + 3 slack, no artificials
    assert std.rows_needing_artificial() == []
    assert np.all(std.b >= 0)
    # slack block should be an identity matrix, matching the legacy transform_to_std_form
    assert np.allclose(std.A[:, 2:], np.eye(3))


def test_mixed_operators_flag_correct_rows_as_needing_artificial():
    problem = LPProblem(
        c=[1.0, 1.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[1.0, 1.0], op="<=", rhs=10.0),
            Constraint(coeffs=[1.0, 0.0], op=">=", rhs=2.0),
            Constraint(coeffs=[0.0, 1.0], op="=", rhs=3.0),
        ],
    )
    std = standardize(problem)
    assert std.rows_needing_artificial() == [1, 2]
    assert 0 in std.slack_col_for_row  # row 0 (<=) has a slack column
    assert 1 in std.slack_col_for_row  # row 1 (>=) has a surplus column
    assert 2 not in std.slack_col_for_row  # row 2 (=) has no slack/surplus column
    assert np.all(std.b >= 0)


def test_negative_rhs_row_is_flipped():
    problem = LPProblem(
        c=[1.0, 1.0],
        sense="maximize",
        constraints=[
            Constraint(coeffs=[-1.0, -1.0], op="<=", rhs=-4.0),
        ],
    )
    std = standardize(problem)
    assert std.b[0] >= 0
    # <= with negative rhs flips to >=, which needs an artificial variable
    assert std.rows_needing_artificial() == [0]


def test_free_variable_is_split_into_two_columns():
    problem = LPProblem(
        c=[1.0, 1.0],
        sense="maximize",
        constraints=[Constraint(coeffs=[1.0, 1.0], op="<=", rhs=4.0)],
        bounds=[(None, None), (0.0, None)],
    )
    std = standardize(problem)
    assert std.n_decision == 3  # free var split into 2 cols + 1 normal var
    assert std.var_mapping[0].kind == "split"
    assert std.var_mapping[1].kind == "simple"


def test_recover_original_x_for_simple_and_split_and_neg_shift():
    problem = LPProblem(
        c=[1.0, 1.0, 1.0],
        sense="maximize",
        constraints=[Constraint(coeffs=[1.0, 1.0, 1.0], op="<=", rhs=10.0)],
        bounds=[(2.0, None), (None, None), (None, 5.0)],
    )
    std = standardize(problem)
    y = np.zeros(std.n)
    # var 0: simple, shift=2 -> x0 = 2 + y[index]
    y[std.var_mapping[0].index] = 3.0
    # var 1: split -> x1 = y[pos] - y[neg]
    y[std.var_mapping[1].pos_index] = 7.0
    y[std.var_mapping[1].neg_index] = 2.0
    # var 2: neg_shift, shift=5 -> x2 = 5 - y[index]
    y[std.var_mapping[2].index] = 1.0

    x = std.recover_original_x(y)
    assert x == [5.0, 5.0, 4.0]


def test_minimize_objective_is_negated_and_recovered():
    problem = LPProblem(
        c=[2.0, 3.0],
        sense="minimize",
        constraints=[Constraint(coeffs=[1.0, 1.0], op="<=", rhs=4.0)],
    )
    std = standardize(problem)
    assert std.minimize is True
    assert np.allclose(std.c[:2], [-2.0, -3.0])
    # maximized_value is in the negated-objective space; recover_objective flips sign back
    assert std.recover_objective(-10.0) == 10.0


def test_bounded_above_variable_adds_extra_row():
    problem = LPProblem(
        c=[1.0],
        sense="maximize",
        constraints=[Constraint(coeffs=[1.0], op="<=", rhs=100.0)],
        bounds=[(0.0, 5.0)],
    )
    std = standardize(problem)
    # original constraint row + 1 extra row for the upper bound
    assert std.m == 2
