# Simplex Solver

A general-purpose linear programming toolkit in Python, built from scratch
on top of a teaching implementation developed for the course "Optimization"
(4th Semester, OTH Regensburg). Beyond solving LPs, it also supports
re-optimization via dual simplex and post-optimal sensitivity analysis
(shadow prices, RHS/cost ranging).

`simplex_solver/` supports:

- `<=`, `>=`, and `=` constraints, via a two-phase primal simplex method
  with artificial variables
- both **maximize** and **minimize** objectives
- free variables and lower/upper-bounded variables
- negative right-hand sides
- explicit `OPTIMAL` / `INFEASIBLE` / `UNBOUNDED` status reporting
- an anti-cycling pivoting rule (Bland's rule fallback after a configurable
  number of iterations) and an incremental (product-form-of-the-inverse)
  basis update instead of recomputing a full matrix inverse every iteration
- a **dual simplex** engine for warm-starting re-optimization after an RHS
  change, instead of rerunning the full two-phase method from scratch
- **sensitivity analysis** at the optimal basis: shadow prices, and the
  ranges over which a constraint's RHS or a variable's objective
  coefficient can move without changing which basis is optimal

The original teaching implementation -- explicit matrix inversion every
iteration, `<=`-only, maximize-only, no infeasibility detection -- is kept
unmodified in `simplex_solver/legacy_teaching.py` as a reference/comparison
baseline. See `simplex.ipynb` for a runnable demo of all of the above.

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

### Sensitivity analysis (shadow prices, RHS/cost ranging)

Pass `compute_sensitivity=True` to get a `SensitivityReport` attached to
the result:

```python
result = solve(problem, compute_sensitivity=True)
print(result.sensitivity.shadow_prices)  # {constraint index -> d(objective)/d(rhs)}
print(result.sensitivity.rhs_ranges)     # {constraint index -> (lb, ub) for that RHS}
print(result.sensitivity.cost_ranges)    # {standard-form column -> (lb, ub) for that cost coeff}
```

`shadow_prices[i]` is the rate of change of the objective per unit
increase in constraint `i`'s right-hand side -- e.g. `0.25` means loosening
that constraint by 1 unit improves the objective by `0.25`, as long as the
new RHS stays inside `rhs_ranges[i]` (outside that range, the optimal
*basis* itself would change, and the shadow price no longer applies as-is).

### Dual simplex: warm-starting after an RHS change

Re-optimizing after tightening/loosening a constraint's RHS doesn't need a
full Phase I/Phase II rerun if the previously-optimal basis is still
dual-feasible (this is the textbook dual-simplex scenario):

```python
from simplex_solver.standardize import standardize
from simplex_solver.phase1 import solve_phase1
from simplex_solver.simplex_core import run_simplex
from simplex_solver.solver import resolve_after_rhs_change

std = standardize(problem)
phase1_result = solve_phase1(std)
phase2_result = run_simplex(
    std.A, std.b, std.c,
    phase1_result.basis_indices, phase1_result.non_basis_indices,
    disallowed_entering=frozenset(std.artificial_col_for_row.values()),
)

new_b = std.b.copy()
new_b[0] -= 3  # tighten the first constraint's RHS
warm_result = resolve_after_rhs_change(
    std, phase2_result.basis_indices, phase2_result.non_basis_indices, new_b,
)
```

See `simplex.ipynb` for this end to end, including an iteration-count
comparison against solving the changed problem from scratch.

## Package layout

```
simplex_solver/
  problem.py          LPProblem / Constraint model + JSON (de)serialization
  standardize.py       General LP -> standard form (>=0 RHS, all vars >=0, max)
  simplex_core.py       Anti-cycling primal pivoting engine (Bland's rule, PFI updates)
  dual_simplex.py       Dual simplex engine for RHS-change re-optimization
  phase1.py            Two-phase method Phase I (artificial variables, infeasibility)
  solver.py            solve() orchestration + resolve_after_rhs_change() warm start
  sensitivity.py         Shadow prices, RHS ranging, cost ranging at the optimal basis
  trace_export.py        JSON export of a solved problem + its iteration trace
  legacy_teaching.py    Original teaching implementation, kept as a baseline
tests/
  fixtures/*.json       Example LPs (canonical, mixed constraints, infeasible, ...)
  test_*.py             Unit, fixture, scipy cross-check, and legacy-comparison tests
viz/
  table_viewer.html      Self-contained HTML viewer for a solved problem's pivot trace
simplex.ipynb          Demo notebook
```

## Running the tests

```
python3 -m pytest tests/
```

The suite includes unit tests per module, an end-to-end fixture suite
cross-checked against `scipy.optimize.linprog` (objectives, and for
sensitivity analysis, shadow prices too), and a comparison test verifying
the new solver agrees with `legacy_teaching.py` on problems both can handle.

## Tech stack

Python, NumPy, pytest (SciPy used only for cross-checking test results).
