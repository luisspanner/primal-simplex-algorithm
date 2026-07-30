"""Verbatim port of the original teaching-style Primal Simplex implementation.

Developed as part of the 4th Semester course "Optimization".

Note:
Expected input is a .txt file that is structured as follows:
- First row: Amount of constraints m and variables n (i. e. 3 2)
- Next exactly m rows: constraints (i. e. 1 1 <= 4)
- Last row: objective function coefficients c (i. e. 1 4)
- objective function is always maximized; if you want to minimize instead, simply turn your minimization problem into a maximization problem

This module is kept unmodified (explicit matrix inversion every iteration,
Phase II only, no Bland's rule) so it can be used as a reference/benchmark
against the general-purpose solver in simplex_solver.solver.
"""
import numpy as np


# Parse input file to extract constraint matrix A, right-hand side vector b, and objective function coefficients c
# handle potential formatting issues and validate the input
def parse_lp(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Parse first line to get number of constraints (m) and number of variables (n)
    first_line = lines[0].strip().split()
    m = int(first_line[0])
    n = int(first_line[1])

    if len(lines) != m + 2:
        raise ValueError(f"Invalid number of lines in the input file. Expected {m+2}, got {len(lines)}.")

    A = np.zeros((m, n))
    b = np.zeros(m)

    # Parse constraints
    for i in range(1, m + 1):
        line = lines[i].strip()

        # Check if the line contains '<='
        if '<=' not in line:
            raise ValueError(f"Invalid constraint format in line {i+1}. Expected '<=. Operator not found.")
        lhs, rhs = line.split('<=', 1) # Split the constraint line into left-hand side and right-hand side
        coeffs = lhs.strip().split() # Split the left-hand side into coefficients
        if len(coeffs) != n:
            raise ValueError(f"Invalid number of coefficients in line {i+1}. Expected {n}, got {len(coeffs)}.")
        A[i-1] = [float(coeff) for coeff in coeffs]
        b[i-1] = float(rhs.strip())

    # Parse objective function
    c = np.array([float(coeff) for coeff in lines[m + 1].strip().split()])

    return A, b, c


# Transform the linear program into standard form by introducing slack variables for each constraint
# Ax + Is = b, x, s ≥ 0, where I is the identity matrix corresponding to the slack variables
def transform_to_std_form(A, b, c):
    m, n = A.shape
    I = np.eye(m) # Identity matrix for slack variables
    A_std = np.hstack((A, I)) # Combine original constraint matrix with identity matrix to form the new constraint matrix
    c_std = np.hstack((c, np.zeros(m))) # Include zero coefficients for the slack variables in the objective function
    return A_std, b, c_std


# get initial basic solution, i. e. set all original variables and use slack variables as a start basis
def get_initial_basic_sol(A, b, c, m, n):
    # Split A into A_B and A_N
    A_B = A[:, n:] # Basic variables correspond to the slack variables
    A_N = A[:, :n] # Non-basic variables correspond to the original variables

    # Split c into c_B and c_N
    c_B = np.zeros(m) # Slack variables are initially in the basis, so their coefficients are zero
    c_N = c[:n] # Original variables are non-basic

    x_B = b.copy() # Initial basic solution is given by the right-hand side vector b
    x_N = np.zeros(n) # Non-basic variables are initially set to zero

    return A_B, A_N, c_B, c_N, x_B, x_N, m, n # return m, n for debugging purposes


# present problem in dictionary form, i. e. express basic variables in terms of non-basic variables
def to_dict(A_B, A_N, c_B, c_N, x_B, x_N, b):
    # compute inverse of A_B
    A_B_inv = np.linalg.inv(A_B)
    # express basic variables in  terms of non-basic variables
    x_B = A_B_inv @ b - A_B_inv @ A_N @ x_N
    # compute the objective function value in terms of non-basic variables
    z = c_B @ A_B_inv @ b + (c_N - c_B @ A_B_inv @ A_N) @ x_N
    return x_B, z


# actual simplex step: compute the tableau, identify entering and leaving variables, perform pivoting to update the basis
def simplex_step(A_B, A_N, c_B, c_N, x_B, x_N, b, basis_indices, non_basis_indices):
    A_B_inv = np.linalg.inv(A_B)
    # maximization problem: compute the reduced costs for non-basic variables
    reduced_costs = c_N - c_B @ A_B_inv @ A_N
    # identify entering variable (most positive reduced cost)
    entering_index = np.argmax(reduced_costs)
    # Optimality check: if all reduced costs are non-positive, we have found the optimal solution
    if reduced_costs[entering_index] <= 0:
        # compute the optimal solution and objective value
        x_B_opt, z = to_dict(A_B, A_N, c_B, c_N, x_B, x_N, b)
        # return optimal solution, objective value and a flag indicating optimality
        return A_B, A_N, c_B, c_N, x_B_opt, x_N, True, basis_indices, non_basis_indices
    # compute direction d: How do the basic variables change as we increase the entering variable?
    d = A_B_inv @ A_N[:, entering_index]
    # Ratio Test: Identify leaving variable
    # for every basic variable, compute how much we can increase the entering variable before any basic variable becomes negative
    deltas = np.full_like(x_B, np.inf) # Initialize ratios with infinity
    for i in range(len(x_B)):
        # only consider basic variables that decrease as we increase the entering variable
        # --> those with positive d[i] since x_B[i] - d[i] * x_N[entering_index] >= 0 must hold to maintain feasibility
        if d[i] > 0:
            # how much can we maximally increase entering variable before x_B[i] becomes zero?
            deltas[i] = x_B[i] / d[i]

    # Check for unboundedness
    if np.all(d <= 0):
        raise ValueError("Linear program is unbounded.")
    # smallest delta indicates the leaving variable
    leaving_index = np.argmin(deltas)

    # swap basis indices
    basis_indices[leaving_index], non_basis_indices[entering_index] = non_basis_indices[entering_index], basis_indices[leaving_index]

    # Pivoting
    old_basis_col = A_B[:, leaving_index].copy() # save old basis col that leaves
    pivot_col = A_N[:, entering_index]
    A_B[:, leaving_index] = pivot_col
    A_N[:, entering_index] = old_basis_col
    old_c_B = c_B[leaving_index].copy() # save old cost coefficient that leaves
    c_B[leaving_index] = c_N[entering_index]
    c_N[entering_index] = old_c_B
    # calculate new inverse of A_B after pivoting
    A_B_inv = np.linalg.inv(A_B)
    x_B = A_B_inv @ b # - A_B_inv @ A_N @ x_N is zero since x_N is zero after pivoting
    x_N = np.zeros(len(c_N)) # non basic variables remain zero
    return A_B, A_N, c_B, c_N, x_B, x_N, False, basis_indices, non_basis_indices


# main function to run the simplex algorithm
def simplex(filename):
    A, b, c = parse_lp(filename)
    A_std, b_std, c_std = transform_to_std_form(A, b, c)
    m, n = A.shape
    A_B, A_N, c_B, c_N, x_B, x_N, m, n = get_initial_basic_sol(A_std, b_std, c_std, m, n)

    basis_indices = list(range(n, n + m)) # initial basis corresponds to slack variables
    non_basis_indices = list(range(n)) # initial non-basis corresponds to original variables
    optimal = False

    # Debugging
    print("A_B shape:", A_B.shape)
    print("A_N shape:", A_N.shape)
    print("m =", m, ", n =", n)

    while not optimal:
        A_B, A_N, c_B, c_N, x_B, x_N, optimal, basis_indices, non_basis_indices = simplex_step(A_B, A_N, c_B, c_N, x_B, x_N, b_std, basis_indices, non_basis_indices)

    return x_B, x_N, c_B @ x_B + c_N @ x_N, optimal, basis_indices, non_basis_indices


if __name__ == "__main__":
    filename = "vl.txt" # specify the input file containing the linear program
    x_B, x_N, obj_value, optimal, basis_indices, non_basis_indices = simplex(filename)

    print("Optimal solution:")
    print(f"Basic variables (indices {basis_indices}): {x_B}")
    print(f"Non-basic variables (indices {non_basis_indices}): {x_N}")
    print(f"Objective value: {obj_value}")
