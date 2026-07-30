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
from simplex_solver.trace_export import export_trace_json

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


def inject_into_html(data, html_path):
    with open(html_path, "r") as f:
        html = f.read()

    start_marker = "/* DATA START */"
    end_marker = "/* DATA END */"
    start = html.index(start_marker) + len(start_marker)
    end = html.index(end_marker)

    payload = f"\nconst TRACE_DATA = {json.dumps(data, indent=2)};\n"
    new_html = html[:start] + payload + html[end:]

    with open(html_path, "w") as f:
        f.write(new_html)


if __name__ == "__main__":
    data = build_data()
    inject_into_html(data, os.path.join(VIZ_DIR, "table_viewer.html"))
    print(f"Injected trace data for {len(data)} fixtures into table_viewer.html")
