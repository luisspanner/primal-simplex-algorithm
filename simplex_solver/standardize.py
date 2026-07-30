"""Transform a general LPProblem into standard form for the simplex core.

Standard form here means: maximize c^T y subject to A y = b, y >= 0, b >= 0.

Handles, beyond the original teaching implementation's Ax <= b / b >= 0 /
max-only / all-vars-x>=0 assumption:
  - minimize objectives (internally negated, un-negated when reporting)
  - >=, <=, = constraints (surplus/slack + artificial variables as needed)
  - negative RHS (row flipped so RHS >= 0)
  - free variables (split x = y_pos - y_neg)
  - variables with a non-zero lower bound (shifted x = y + lb)
  - variables with a finite upper bound (extra "<=" row, or for
    free-below/bounded-above vars, x = ub - y so y >= 0 enforces x <= ub)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from simplex_solver.problem import Constraint, LPProblem


@dataclass
class VarSplit:
    """How one original variable maps onto standard-form columns."""
    kind: str  # "simple" | "neg_shift" | "split"
    index: int = -1       # "simple"/"neg_shift": standard-form column index
    shift: float = 0.0    # constant added back: x = shift + sign(s) * y
    pos_index: int = -1   # "split": column for y_pos
    neg_index: int = -1   # "split": column for y_neg

    def recover(self, y: np.ndarray) -> float:
        if self.kind == "simple":
            return self.shift + y[self.index]
        if self.kind == "neg_shift":
            return self.shift - y[self.index]
        if self.kind == "split":
            return y[self.pos_index] - y[self.neg_index]
        raise ValueError(f"Unknown VarSplit kind {self.kind!r}")


@dataclass
class StandardForm:
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    n_decision: int
    slack_col_for_row: Dict[int, int]
    artificial_col_for_row: Dict[int, int]
    var_mapping: List[VarSplit]
    minimize: bool
    objective_shift: float

    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]

    def rows_needing_artificial(self) -> List[int]:
        return sorted(self.artificial_col_for_row.keys())

    def recover_original_x(self, y: np.ndarray) -> List[float]:
        return [split.recover(y) for split in self.var_mapping]

    def recover_objective(self, maximized_value: float) -> float:
        raw = maximized_value + self.objective_shift
        return -raw if self.minimize else raw


def _build_var_mapping(problem: LPProblem):
    """Returns (var_mapping, n_decision, extra_rows) where extra_rows is a
    list of (col_index, op, rhs) for variables with a finite upper bound
    that need an explicit "<=" row on top of the column substitution."""
    var_mapping: List[VarSplit] = []
    extra_rows: List[Tuple[int, str, float]] = []
    n_decision = 0

    for lb, ub in problem.effective_bounds():
        if lb is None and ub is None:
            pos_index, neg_index = n_decision, n_decision + 1
            n_decision += 2
            var_mapping.append(VarSplit(kind="split", pos_index=pos_index, neg_index=neg_index))
        elif lb is None and ub is not None:
            index = n_decision
            n_decision += 1
            var_mapping.append(VarSplit(kind="neg_shift", index=index, shift=ub))
        else:
            index = n_decision
            n_decision += 1
            shift = 0.0 if lb is None else lb
            var_mapping.append(VarSplit(kind="simple", index=index, shift=shift))
            if ub is not None:
                extra_rows.append((index, "<=", ub - shift))

    return var_mapping, n_decision, extra_rows


def _transform_row(coeffs, rhs, var_mapping, n_decision):
    """Map original-variable coefficients onto standard-form decision
    columns, folding each variable's constant shift into the RHS."""
    row = np.zeros(n_decision)
    rhs_adjustment = 0.0
    for coeff, split in zip(coeffs, var_mapping):
        if split.kind == "simple":
            row[split.index] += coeff
            rhs_adjustment += coeff * split.shift
        elif split.kind == "neg_shift":
            row[split.index] += -coeff
            rhs_adjustment += coeff * split.shift
        elif split.kind == "split":
            row[split.pos_index] += coeff
            row[split.neg_index] += -coeff
    return row, rhs - rhs_adjustment


def standardize(problem: LPProblem) -> StandardForm:
    problem.validate()
    minimize = problem.sense == "minimize"
    working_c = [-x for x in problem.c] if minimize else list(problem.c)

    var_mapping, n_decision, extra_rows = _build_var_mapping(problem)

    c_decision, objective_shift = _transform_row(working_c, 0.0, var_mapping, n_decision)
    objective_shift = -objective_shift  # _transform_row returns (row, rhs - adjustment); rhs=0 here

    all_rows: List[Tuple[np.ndarray, str, float]] = []
    for constraint in problem.constraints:
        row, rhs = _transform_row(constraint.coeffs, constraint.rhs, var_mapping, n_decision)
        all_rows.append((row, constraint.op, rhs))
    for col_index, op, rhs in extra_rows:
        row = np.zeros(n_decision)
        row[col_index] = 1.0
        all_rows.append((row, op, rhs))

    # Flip rows with negative RHS so every row ends up with rhs >= 0.
    flipped_rows = []
    for row, op, rhs in all_rows:
        if rhs < 0:
            row = -row
            rhs = -rhs
            if op == "<=":
                op = ">="
            elif op == ">=":
                op = "<="
        flipped_rows.append((row, op, rhs))

    m = len(flipped_rows)
    slack_col_for_row: Dict[int, int] = {}
    artificial_col_for_row: Dict[int, int] = {}
    n_slack_surplus = sum(1 for _, op, _ in flipped_rows if op in ("<=", ">="))
    n_artificial = sum(1 for _, op, _ in flipped_rows if op in ("=", ">="))
    n_std = n_decision + n_slack_surplus + n_artificial

    A = np.zeros((m, n_std))
    b = np.zeros(m)
    next_slack_col = n_decision
    next_artificial_col = n_decision + n_slack_surplus

    for i, (row, op, rhs) in enumerate(flipped_rows):
        A[i, :n_decision] = row
        b[i] = rhs
        if op == "<=":
            slack_col_for_row[i] = next_slack_col
            A[i, next_slack_col] = 1.0
            next_slack_col += 1
        elif op == ">=":
            slack_col_for_row[i] = next_slack_col
            A[i, next_slack_col] = -1.0
            next_slack_col += 1
            artificial_col_for_row[i] = next_artificial_col
            A[i, next_artificial_col] = 1.0
            next_artificial_col += 1
        elif op == "=":
            artificial_col_for_row[i] = next_artificial_col
            A[i, next_artificial_col] = 1.0
            next_artificial_col += 1
        else:
            raise ValueError(f"Unexpected operator after normalization: {op!r}")

    c = np.zeros(n_std)
    c[:n_decision] = c_decision

    return StandardForm(
        A=A,
        b=b,
        c=c,
        n_decision=n_decision,
        slack_col_for_row=slack_col_for_row,
        artificial_col_for_row=artificial_col_for_row,
        var_mapping=var_mapping,
        minimize=minimize,
        objective_shift=objective_shift,
    )
