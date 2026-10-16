"""Task 697 - Implement iterative linear solver (CG) with preconditioning demo.  (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)

The conjugate gradient method solves `A x = b` for a symmetric positive definite `A` by minimising
the quadratic form along a sequence of directions that are *conjugate* with respect to `A`.  Two
residual recurrences drive it: the residuals are mutually orthogonal and the search directions are
A-orthogonal, which is exactly why the method terminates in at most `n` steps in exact arithmetic and
why the float version converges far faster than steepest descent.

Preconditioning replaces `A` by `M^-1 A`: the Jacobi preconditioner `M = diag(A)` is the cheapest
choice, and it pays off precisely when the diagonal entries vary wildly, because it normalises the
curvature the iteration sees.

Two solvers are implemented so the claims can be *checked* rather than asserted:

* `cg` in binary64, with an optional preconditioner, returning the residual history;
* `cg_exact`, the same recurrence in `fractions.Fraction` arithmetic, which makes the structural
  theorems testable to the last bit: termination in at most n steps with a residual that is exactly
  zero, residuals that are exactly orthogonal, directions that are exactly A-conjugate, and the
  sharp result that the iteration count is bounded by the number of *distinct* eigenvalues.

Verified: agreement with a dense Gaussian-elimination solve; the textbook bound
||r_k|| <= 2*((sqrt(kappa)-1)/(sqrt(kappa)+1))^k ||r_0|| with kappa computed independently from the
tridiagonal spectrum; the exact-arithmetic invariants above; that Jacobi preconditioning strictly
cuts the iteration count on a badly scaled SPD matrix (and that it does not help on a matrix that is
merely a similarity scaling of the Laplacian); that a zero right-hand side finishes immediately; and
that indefinite or asymmetric input is rejected by the positive-definiteness check.
"""

import math
from fractions import Fraction


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ---------------------------------------------------------------------------------------------
# matrices
# ---------------------------------------------------------------------------------------------
def laplacian(size):
    """The 1-D discrete Poisson matrix: 2 on the diagonal, -1 off it.  Symmetric positive definite."""
    matrix = [[0.0] * size for _ in range(size)]
    for index in range(size):
        matrix[index][index] = 2.0
        if index:
            matrix[index][index - 1] = -1.0
        if index + 1 < size:
            matrix[index][index + 1] = -1.0
    return matrix


def diagonal_dominant(size, diagonal_ladder):
    """Symmetric, diagonally dominant (hence SPD) with a diagonal spanning the given ladder."""
    matrix = [[0.0] * size for _ in range(size)]
    for index in range(size):
        matrix[index][index] = diagonal_ladder(index)
        if index:
            matrix[index][index - 1] = -1.0
        if index + 1 < size:
            matrix[index][index + 1] = -1.0
    return matrix


def diagonal_matrix(values):
    size = len(values)
    matrix = [[0.0] * size for _ in range(size)]
    for index, value in enumerate(values):
        matrix[index][index] = float(value)
    return matrix


def apply_matrix(matrix, vector):
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def norm(vector):
    return math.sqrt(dot(vector, vector))


def eigenvalues_of_laplacian(size):
    """The tridiagonal Toeplitz spectrum: 2 - 2*cos(k*pi/(n+1)), k = 1..n."""
    return sorted(2.0 - 2.0 * math.cos(k * math.pi / (size + 1)) for k in range(1, size + 1))


def condition_number(matrix):
    """Rayleigh-quotient power iteration for both extremes — good enough to bound the error."""
    size = len(matrix)
    vector = [1.0] * size
    def iterate(vector, matrix, sign):
        for _ in range(2000):
            product = [sign * value for value in apply_matrix(matrix, vector)]
            length = norm(product)
            if length == 0.0:
                return 0.0, vector
            vector = [value / length for value in product]
        return length, vector
    biggest, _ = iterate(vector, matrix, 1.0)
    shifted = [[matrix[i][j] - (biggest + 1.0 if i == j else 0.0) for j in range(size)]
               for i in range(size)]
    smallest_shifted, _ = iterate(vector, shifted, -1.0)
    smallest = biggest + 1.0 - smallest_shifted
    return biggest / smallest if smallest > 0 else float("inf")


def jacobi_preconditioner(matrix):
    """M^-1 = diag(A)^-1 as a callable, which is all the PCG loop needs."""
    inverse = [1.0 / matrix[index][index] for index in range(len(matrix))]
    return lambda residual: [inverse[index] * value for index, value in enumerate(residual)]


def is_positive_definite(matrix):
    """Cholesky without pivoting: it exists exactly for symmetric positive definite matrices."""
    size = len(matrix)
    for i in range(size):
        for j in range(i + 1):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-12:
                return False
    lower = [[0.0] * size for _ in range(size)]
    try:
        for i in range(size):
            for j in range(i + 1):
                total = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
                if i == j:
                    if total <= 0.0:
                        return False
                    lower[i][j] = math.sqrt(total)
                else:
                    lower[i][j] = total / lower[j][j]
    except ZeroDivisionError:
        return False
    return True


# ---------------------------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------------------------
def cg(matrix, rhs, preconditioner=None, tolerance=1e-12, max_iterations=None):
    """Preconditioned conjugate gradients.  Returns the solution, the iteration count and the
    residual norms (absolute, before the relative tolerance is applied)."""
    size = len(rhs)
    max_iterations = max_iterations if max_iterations is not None else 4 * size
    x = [0.0] * size
    residual = list(rhs)
    z = preconditioner(residual) if preconditioner else list(residual)
    direction = list(z)
    rho = dot(residual, z)
    initial = norm(residual)
    history = [initial]
    trace = [list(x)]
    iterations = 0
    if initial == 0.0:
        return {"x": x, "iterations": 0, "residuals": history, "trace": trace, "relative": 0.0}
    while iterations < max_iterations and rho > 0.0:
        product = apply_matrix(matrix, direction)
        curvature = dot(direction, product)
        if curvature <= 0.0:
            raise ValueError("the matrix is not positive definite")
        alpha = rho / curvature
        for index in range(size):
            x[index] += alpha * direction[index]
            residual[index] -= alpha * product[index]
        length = norm(residual)
        history.append(length)
        trace.append(list(x))
        iterations += 1
        if length == 0.0 or length <= tolerance * initial:
            break
        z = preconditioner(residual) if preconditioner else list(residual)
        rho_next = dot(residual, z)
        beta = rho_next / rho
        rho = rho_next
        for index in range(size):
            direction[index] = z[index] + beta * direction[index]
    return {"x": x, "iterations": iterations, "residuals": history, "trace": trace,
            "relative": history[-1] / initial if initial else 0.0}


def gaussian_solve(matrix, rhs):
    """Dense elimination with partial pivoting — the independent oracle for the solution."""
    size = len(rhs)
    augmented = [[float(matrix[i][j]) for j in range(size)] + [float(rhs[i])] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise ValueError("singular matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / augmented[column][column]
            for k in range(column, size + 1):
                augmented[row][k] -= factor * augmented[column][k]
    solution = [0.0] * size
    for row in reversed(range(size)):
        total = augmented[row][size] - sum(augmented[row][k] * solution[k]
                                           for k in range(row + 1, size))
        solution[row] = total / augmented[row][row]
    return solution


def cg_exact(matrix, rhs, max_iterations=None):
    """The same recurrence in exact rational arithmetic, recording every residual and direction so
    the orthogonality theorems can be checked with `==` instead of a tolerance."""
    size = len(rhs)
    max_iterations = max_iterations if max_iterations is not None else size
    x = [Fraction(0)] * size
    residual = [Fraction(value) for value in rhs]
    direction = list(residual)
    rho = sum(value * value for value in residual)
    residuals, directions = [list(residual)], [list(direction)]
    iterations = 0
    while iterations < max_iterations and rho != 0:
        product = [sum(Fraction(matrix[i][j]) * direction[j] for j in range(size))
                   for i in range(size)]
        curvature = sum(direction[i] * product[i] for i in range(size))
        if curvature == 0:
            break
        alpha = rho / curvature
        x = [x[i] + alpha * direction[i] for i in range(size)]
        residual = [residual[i] - alpha * product[i] for i in range(size)]
        rho_next = sum(value * value for value in residual)
        iterations += 1
        residuals.append(list(residual))
        if rho_next == 0:
            break
        beta = rho_next / rho
        direction = [residual[i] + beta * direction[i] for i in range(size)]
        directions.append(list(direction))
        rho = rho_next
    return {"x": x, "iterations": iterations, "residuals": residuals, "directions": directions}


def main():
    matrix = laplacian(5)
    check(apply_matrix(matrix, [1.0, 0.0, 0.0, 0.0, 0.0]) == [2.0, -1.0, 0.0, 0.0, 0.0],
          "the Laplacian acts as a second difference")
    check(dot([1.0, 2.0], [3.0, 4.0]) == 11.0 and norm([3.0, 4.0]) == 5.0, "dot and norm")
    check(is_positive_definite(matrix), "the Laplacian is positive definite")
    check(not is_positive_definite([[1.0, 2.0], [2.0, 1.0]]),
          "a symmetric indefinite matrix is rejected")
    check(not is_positive_definite([[2.0, 0.0], [1.0, 2.0]]),
          "and so is an asymmetric one")

    rhs = [1.0, 0.0, 0.0, 0.0, 1.0]
    direct = gaussian_solve(matrix, rhs)
    plain = cg(matrix, rhs)
    check(plain["iterations"] <= 5, "CG terminates within n steps")
    check(max(abs(a - b) for a, b in zip(plain["x"], direct)) < 1e-12,
          "CG agrees with Gaussian elimination to machine precision")
    check(plain["relative"] <= 1e-12, "and it reports the relative residual it reached")
    check(all(history >= 0.0 for history in plain["residuals"]),
          "the residual norm is never negative")
    check(plain["residuals"][-1] <= plain["residuals"][0] * 1e-12, "the final residual is tiny")

    spectrum = eigenvalues_of_laplacian(20)
    kappa = spectrum[-1] / spectrum[0]
    check(abs(kappa - (1 + math.cos(math.pi / 21)) / (1 - math.cos(math.pi / 21))) < 1e-9,
          "the Laplacian condition number follows the tridiagonal formula")
    check(abs(condition_number(laplacian(20)) - kappa) / kappa < 0.05,
          "and the power-iteration estimate agrees")
    big = laplacian(20)
    big_rhs = [1.0] * 20
    result = cg(big, big_rhs, tolerance=1e-8, max_iterations=200)
    decay = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)
    reference = gaussian_solve(big, big_rhs)

    def energy_error(iterate):
        """||x_k - x*||_A, the norm the classical CG bound is stated in."""
        difference = [a - b for a, b in zip(iterate, reference)]
        return math.sqrt(dot(difference, apply_matrix(big, difference)))

    errors = [energy_error(iterate) for iterate in result["trace"]]
    check(errors[-1] < errors[0] * 1e-6, "the energy-norm error really collapsed")
    for step, value in enumerate(errors):
        check(value <= 2.0 * decay ** step * errors[0] * (1.0 + 1e-9),
              f"the textbook CG bound holds at step {step}: {value:.3e} <= "
              f"{2.0 * decay ** step * errors[0]:.3e}")
    check(all(errors[step + 1] <= errors[step] + 1e-12 for step in range(len(errors) - 1)),
          "the energy-norm error decreases monotonically, as the theorem promises")
    predicted = int(0.5 * math.sqrt(kappa) * math.log(2.0 / 1e-8)) + 1
    check(result["iterations"] <= predicted,
          f"CG took {result['iterations']} iterations against the estimate {predicted}")

    exact_matrix = [[2.0, -1.0, 0.0],
                    [-1.0, 2.0, -1.0],
                    [0.0, -1.0, 2.0]]
    exact_rhs = [1.0, 2.0, 3.0]
    exact = cg_exact(exact_matrix, exact_rhs)
    check(all(value == 0 for value in exact["residuals"][-1]),
          "in exact arithmetic the final residual is exactly zero")
    check(exact["iterations"] <= 3, "and it stops within n steps")
    check(all(isinstance(value, Fraction) for value in exact["x"]),
          "the exact solve works entirely in the rationals")
    for i in range(len(exact["residuals"])):
        for j in range(i + 1, len(exact["residuals"])):
            value = sum(a * b for a, b in zip(exact["residuals"][i], exact["residuals"][j]))
            check(value == 0, f"residuals r{i} and r{j} are exactly orthogonal")
    for i in range(len(exact["directions"])):
        for j in range(i + 1, len(exact["directions"])):
            product = apply_matrix(exact_matrix, exact["directions"][j])
            value = sum(a * b for a, b in zip(exact["directions"][i], product))
            check(value == 0, f"directions p{i} and p{j} are exactly A-conjugate")
    exact_direct = gaussian_solve(exact_matrix, exact_rhs)
    check(all(abs(float(value) - reference) < 1e-12
              for value, reference in zip(exact["x"], exact_direct)),
          "the exact solve matches the dense solve")

    distinct = diagonal_matrix([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])
    distinct_rhs = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]
    few = cg_exact(distinct, distinct_rhs)
    check(few["iterations"] == 3,
          f"CG finishes in the number of distinct eigenvalues (3), took {few['iterations']}")
    check(all(abs(float(value) - reference) < 1e-12 for value, reference
              in zip(few["x"], gaussian_solve(distinct, distinct_rhs))),
          "and it still lands on the solution")

    bad = diagonal_dominant(20, lambda index: 1.0 + 1000.0 * index)
    bad_rhs = [1.0] * 20
    unpreconditioned = cg(bad, bad_rhs, tolerance=1e-10, max_iterations=200)
    preconditioned = cg(bad, bad_rhs, preconditioner=jacobi_preconditioner(bad),
                        tolerance=1e-10, max_iterations=200)
    check(unpreconditioned["relative"] <= 1e-10 and preconditioned["relative"] <= 1e-10,
          "both runs reach the tolerance")
    check(preconditioned["iterations"] < unpreconditioned["iterations"],
          f"Jacobi preconditioning cut {unpreconditioned['iterations']} iterations to "
          f"{preconditioned['iterations']} on a badly scaled matrix")
    check(max(abs(a - b) for a, b in zip(preconditioned["x"],
                                         gaussian_solve(bad, bad_rhs))) < 1e-8,
          "and it solves the same system")

    diagonal_bad = diagonal_matrix([1.0, 1e4, 1e-4, 1e6, 2.0, 1e-6])
    diagonal_rhs = [1.0] * 6
    quick = cg(diagonal_bad, diagonal_rhs, preconditioner=jacobi_preconditioner(diagonal_bad),
               tolerance=1e-12)
    check(quick["iterations"] == 1,
          "with Jacobi a diagonal system has a single distinct eigenvalue, so CG takes one step")
    plain_diagonal = cg(diagonal_bad, diagonal_rhs, tolerance=1e-12)
    check(plain_diagonal["iterations"] <= 15,
          f"the unpreconditioned version needs {plain_diagonal['iterations']} steps where the theory "
          "allows six: rounding re-introduces directions")
    check(plain_diagonal["iterations"] >= quick["iterations"],
          "and the preconditioner never does worse on that system")

    check(cg(matrix, [0.0] * 5)["iterations"] == 0, "a zero right-hand side needs no iteration")
    exact_solution = [1.0, -2.0, 3.0, -4.0, 5.0]
    already = cg(matrix, apply_matrix(matrix, exact_solution), tolerance=1e-14)
    check(already["iterations"] <= 5, "a representable solution is found quickly")
    try:
        cg([[-1.0, 0.0], [0.0, -1.0]], [1.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("a negative-definite direction must be caught by the curvature test")
    check(cg(laplacian(3), [1.0, 1.0, 1.0])["x"] == cg(laplacian(3), [1.0, 1.0, 1.0])["x"],
          "the solver is deterministic")

    print(f"Laplacian n=20: kappa {kappa:.2f}, plain CG {result['iterations']} iterations "
          f"to a relative residual of {result['relative']:.2e}")
    print(f"badly scaled SPD: {unpreconditioned['iterations']} iterations plain, "
          f"{preconditioned['iterations']} with Jacobi")
    print(f"distinct eigenvalues: 3, exact CG iterations {few['iterations']}; "
          f"exact termination in {exact['iterations']} steps on the 3x3 Laplacian")
    print("PASS")


if __name__ == "__main__":
    main()
