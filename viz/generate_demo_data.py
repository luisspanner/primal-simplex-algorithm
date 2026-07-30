"""Generate the embedded trace data for viz/table_viewer.html.

Solves a curated set of fixtures with collect_trace=True, exports each via
trace_export.export_trace_json, and writes the result as a single inline
<script> block into table_viewer.html (between the DATA markers), so the
page stays a single self-contained file with no fetch/CORS concerns.
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from simplex_solver import LPProblem, solve
from simplex_solver.phase1 import solve_phase1
from simplex_solver.simplex_core import SimplexDidNotConverge, run_simplex
from simplex_solver.solver import _to_solver_trace_step
from simplex_solver.standardize import standardize
from simplex_solver.trace_export import _trace_step_dict, export_trace_json

FIXTURES_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")
VIZ_DIR = os.path.dirname(os.path.abspath(__file__))

DEMO_FIXTURES = {
    "vl": ("vl.json", "Canonical <=  (vl.json)"),
    "mixed_ge_eq": ("mixed_ge_eq.json", "Mixed <=, =, >=  (mixed_ge_eq.json)"),
    "minimize": ("minimize.json", "Minimize, mixed >=/=  (minimize.json)"),
    "degenerate_beale": ("degenerate_beale.json", "Beale's cycling example  (degenerate_beale.json)"),
}


def build_data():
    data = {}
    for key, (filename, label) in DEMO_FIXTURES.items():
        problem = LPProblem.load_json(os.path.join(FIXTURES_DIR, filename))
        result = solve(problem, collect_trace=True)
        exported = export_trace_json(problem, result)
        exported["label"] = label
        data[key] = exported
    return data


def build_comparison_data():
    """Beale's cycling example, run to completion under plain Dantzig's
    rule (which cycles forever -- capped at 30 iterations to show the
    repeating pattern) vs. Bland's rule from iteration 0 (converges)."""
    problem = LPProblem.load_json(os.path.join(FIXTURES_DIR, "degenerate_beale.json"))
    std = standardize(problem)
    phase1_result = solve_phase1(std)  # trivial fast path: all <=, rhs >= 0

    try:
        run_simplex(
            std.A, std.b, std.c,
            phase1_result.basis_indices, phase1_result.non_basis_indices,
            bland_after=10_000, max_iterations=30, collect_trace=True,
        )
        raise AssertionError("expected Beale's example to cycle under plain Dantzig's rule")
    except SimplexDidNotConverge as exc:
        dantzig_trace = [
            _trace_step_dict(_to_solver_trace_step(std, step, phase=2))
            for step in exc.trace
        ]

    bland_result = run_simplex(
        std.A, std.b, std.c,
        phase1_result.basis_indices, phase1_result.non_basis_indices,
        bland_after=0, max_iterations=100, collect_trace=True,
    )
    bland_trace = [
        _trace_step_dict(_to_solver_trace_step(std, step, phase=2))
        for step in bland_result.trace
    ]

    return {
        "problem": problem.to_dict(),
        "dantzig": {
            "converged": False,
            "trace": dantzig_trace,
            "note": "Plain Dantzig's rule (most-positive reduced cost) cycles "
                    "forever on this classic example -- shown capped at 30 "
                    "iterations; watch the basis columns repeat.",
        },
        "bland": {
            "converged": True,
            "trace": bland_trace,
            "note": "Switching to Bland's rule (smallest index, both entering "
                    "and leaving) from the very first iteration guarantees "
                    "termination.",
        },
    }


def inject_into_html(data, comparison_data, html_path):
    with open(html_path, "r") as f:
        html = f.read()

    def replace_between(html, start_marker, end_marker, payload):
        start = html.index(start_marker) + len(start_marker)
        end = html.index(end_marker)
        return html[:start] + payload + html[end:]

    html = replace_between(
        html, "/* DATA START */", "/* DATA END */",
        f"\nconst TRACE_DATA = {json.dumps(data, indent=2)};\n",
    )
    html = replace_between(
        html, "/* COMPARISON START */", "/* COMPARISON END */",
        f"\nconst COMPARISON_DATA = {json.dumps(comparison_data, indent=2)};\n",
    )

    with open(html_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    data = build_data()
    comparison_data = build_comparison_data()
    inject_into_html(data, comparison_data, os.path.join(VIZ_DIR, "table_viewer.html"))
    print(f"Injected trace data for {len(data)} fixtures + Dantzig/Bland comparison into table_viewer.html")
