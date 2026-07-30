# Simplex Solver

A general-purpose Simplex-method linear programming solver in Python, built
on top of a teaching implementation developed for the course "Optimization"
(4th Semester, OTH Regensburg).

`simplex_solver/` supports:

- `<=`, `>=`, and `=` constraints, via a two-phase method with artificial
  variables
- both **maximize** and **minimize** objectives
- free variables and lower/upper-bounded variables
- negative right-hand sides
- explicit `OPTIMAL` / `INFEASIBLE` / `UNBOUNDED` status reporting
- an anti-cycling pivoting rule (Bland's rule fallback after a configurable
  number of iterations) and an incremental (product-form-of-the-inverse)
  basis update instead of recomputing a full matrix inverse every iteration

The original teaching implementation -- explicit matrix inversion every
iteration, `<=`-only, maximize-only, no infeasibility detection -- is kept
unmodified in `simplex_solver/legacy_teaching.py` as a reference/comparison
baseline. See `simplex.ipynb` for a runnable demo of both, side by side.

## Usage

```python
from simplex_solver import solve, LPProblem, Constraint, Status

problem = LPProblem(
    c=[3, 2],
    sense="maximize",
    constraints=[
        Constraint(coeffs=[4, 2], op="<=", rhs=9),
        Constraint(coeffs=[10, 20], op="<=", rhs=51),
        Constraint(coeffs=[4, 3], op="<=", rhs=10),
    ],
)
result = solve(problem)
print(result.status)            # Status.OPTIMAL
print(result.x)                 # [1.75, 1.0]
print(result.objective_value)   # 7.25
```

`LPProblem` can also be loaded from/dumped to JSON:

```python
problem = LPProblem.load_json("tests/fixtures/vl.json")
```

Variable bounds default to `x >= 0`. To declare a free or custom-bounded
variable, pass `bounds=[(lb, ub), ...]` (one `(lb, ub)` pair per variable,
`None` for unbounded in that direction) when constructing `LPProblem`.

## Package layout

```
simplex_solver/
  problem.py          LPProblem / Constraint model + JSON (de)serialization
  standardize.py       General LP -> standard form (>=0 RHS, all vars >=0, max)
  simplex_core.py       Anti-cycling pivoting engine (Bland's rule, PFI updates)
  phase1.py            Two-phase method Phase I (artificial variables, infeasibility)
  solver.py            solve() orchestration: standardize -> Phase I -> Phase II
  legacy_teaching.py    Original teaching implementation, kept as a baseline
tests/
  fixtures/*.json       Example LPs (canonical, mixed constraints, infeasible, ...)
  test_*.py             Unit, fixture, scipy cross-check, and legacy-comparison tests
simplex.ipynb          Demo notebook
```

## Running the tests

```
python3 -m pytest tests/
```

The suite includes unit tests per module, an end-to-end fixture suite
cross-checked against `scipy.optimize.linprog`, and a comparison test
verifying the new solver agrees with `legacy_teaching.py` on problems both
can handle.

## Tech stack

Python, NumPy, pytest (SciPy used only for cross-checking test results).
