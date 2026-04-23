# Primal Simplex Algorithm for Linear Optimization

Implementation of the primal Simplex method for linear optimization problems 
from scratch, developed as part of the course "Optimization" (4th Semester, 
OTH Regensburg).

## Overview

The algorithm solves linear programs of the form:

maximize c^T x subject to Ax ≤ b, x ≥ 0

## Features

- File-based input parsing for linear programs
- Automatic transformation to standard form (slack variables)
- Dictionary representation of the LP
- Pivot selection via most positive reduced cost
- Ratio test for leaving variable selection
- Unboundedness detection

## Usage

Input is a `.txt` file structured as follows:
- First line: number of constraints `m` and variables `n`
- Next `m` lines: constraints in the form `coeff1 coeff2 ... <= rhs`
- Last line: objective function coefficients `c`

See `test_lp.txt` for an example. The algorithm always maximizes — 
for minimization, convert manually.

Then run the notebook and set the filename accordingly.

## Tech Stack

Python, NumPy
