"""General LP problem model with JSON (de)serialization and validation.

JSON shape:
{
  "sense": "maximize" | "minimize",
  "c": [<float>, ...],
  "constraints": [
    {"coeffs": [<float>, ...], "op": "<=" | ">=" | "=", "rhs": <float>},
    ...
  ],
  "bounds": [[<float-or-null lb>, <float-or-null ub>], ...]   // optional
}

`bounds` defaults to (0, None) (i.e. x >= 0, unbounded above) per variable
when omitted. A null lower bound means the variable is free/unrestricted.
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

VALID_OPS = {"<=", ">=", "="}
VALID_SENSES = {"maximize", "minimize"}


@dataclass
class Constraint:
    coeffs: List[float]
    op: str
    rhs: float

    def to_dict(self):
        return {"coeffs": list(self.coeffs), "op": self.op, "rhs": self.rhs}

    @staticmethod
    def from_dict(data):
        if "coeffs" not in data or "op" not in data or "rhs" not in data:
            raise ValueError(f"Constraint missing required field(s): {data}")
        return Constraint(
            coeffs=[float(x) for x in data["coeffs"]],
            op=data["op"],
            rhs=float(data["rhs"]),
        )


@dataclass
class LPProblem:
    c: List[float]
    sense: str
    constraints: List[Constraint]
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None

    @property
    def n_vars(self) -> int:
        return len(self.c)

    def effective_bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        if self.bounds is None:
            return [(0.0, None) for _ in range(self.n_vars)]
        return self.bounds

    def validate(self) -> None:
        if self.sense not in VALID_SENSES:
            raise ValueError(f"Invalid sense {self.sense!r}, expected one of {VALID_SENSES}")

        n = self.n_vars
        if n == 0:
            raise ValueError("Objective coefficients 'c' must be non-empty")

        for i, constraint in enumerate(self.constraints):
            if constraint.op not in VALID_OPS:
                raise ValueError(
                    f"Constraint {i}: invalid operator {constraint.op!r}, expected one of {VALID_OPS}"
                )
            if len(constraint.coeffs) != n:
                raise ValueError(
                    f"Constraint {i}: expected {n} coefficients, got {len(constraint.coeffs)}"
                )

        bounds = self.bounds
        if bounds is not None:
            if len(bounds) != n:
                raise ValueError(f"Expected {n} variable bounds, got {len(bounds)}")
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None and ub is not None and lb > ub:
                    raise ValueError(f"Variable {i}: lower bound {lb} exceeds upper bound {ub}")

    def to_dict(self):
        result = {
            "sense": self.sense,
            "c": list(self.c),
            "constraints": [c.to_dict() for c in self.constraints],
        }
        if self.bounds is not None:
            result["bounds"] = [list(b) for b in self.bounds]
        return result

    @staticmethod
    def from_dict(data) -> "LPProblem":
        if "c" not in data or "sense" not in data or "constraints" not in data:
            raise ValueError("Problem JSON must contain 'c', 'sense', and 'constraints'")
        c = [float(x) for x in data["c"]]
        sense = data["sense"]
        constraints = [Constraint.from_dict(cd) for cd in data["constraints"]]
        bounds = None
        if "bounds" in data and data["bounds"] is not None:
            bounds = [tuple(b) for b in data["bounds"]]
        problem = LPProblem(c=c, sense=sense, constraints=constraints, bounds=bounds)
        problem.validate()
        return problem

    def dump_json(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load_json(filename: str) -> "LPProblem":
        with open(filename, "r") as f:
            data = json.load(f)
        return LPProblem.from_dict(data)
