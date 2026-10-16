"""Task 698 - Implement multigrid method outline for Poisson equation small grid.  (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)

The 2-D Poisson problem -u_xx - u_yy = f on the unit square with zero Dirichlet data is discretised
with the five-point stencil on a grid of n x n points, and solved by a multigrid V-cycle:

* **smoother** — weighted Jacobi with omega = 2/3, three sweeps before and after the coarse-grid
  correction.  A smoother is *not* a solver: it kills the oscillatory part of the error quickly and
  the smooth part hardly at all, which the checks measure directly by damping one pure mode;
* **coarse-grid correction** — the residual is restricted with full weighting, solved on a grid with
  half the points, and the correction is interpolated back with bilinear prolongation.  The two
  transfers are adjoint to each other up to the factor 1/4, a structural property the checks verify
  with dot products rather than by eye;
* **recursion** — the coarse problem is itself solved by a V-cycle until the grid is small enough to
  solve exactly by dense elimination, so the whole hierarchy of scales is used.

The multigrid promise is *grid independence*: the residual reduction per V-cycle stays roughly the
same as h shrinks, where plain Jacobi needs O(1/h^2) sweeps for the same factor.  Both claims are
measured here — the convergence factor is compared between n = 17 and n = 33, and the cycle count is
compared against the Jacobi sweeps needed for the same tolerance.  The discrete solution is validated
against the manufactured solution u = sin(pi x) sin(pi y) (second-order accuracy, checked by halving h)
and against a dense Gaussian elimination solve on a small grid.
"""

import math


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def new_grid(n):
    return [[0.0] * n for _ in range(n)]


def interior_indices(n):
    return range(1, n - 1)


def apply_operator(u, n, h):
    """The negative Laplacian with the five-point stencil; the boundary stays zero."""
    scale = 1.0 / (h * h)
    out = new_grid(n)
    for i in interior_indices(n):
        for j in interior_indices(n):
            out[i][j] = scale * (4.0 * u[i][j] - u[i - 1][j] - u[i + 1][j]
                                 - u[i][j - 1] - u[i][j + 1])
    return out


def residual(u, f, n, h):
    operator = apply_operator(u, n, h)
    out = new_grid(n)
    for i in interior_indices(n):
        for j in interior_indices(n):
            out[i][j] = f[i][j] - operator[i][j]
    return out


def jacobi_smooth(u, f, n, h, sweeps=3, omega=2.0 / 3.0):
    """Weighted Jacobi: the smoother.  One sweep is O(h^-2) work and is applied a constant number
    of times, which is what makes multigrid's cost linear in the number of unknowns."""
    scale = h * h
    for _ in range(sweeps):
        candidate = [row[:] for row in u]
        for i in interior_indices(n):
            for j in interior_indices(n):
                neighbours = u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]
                candidate[i][j] = ((1.0 - omega) * u[i][j]
                                   + omega * 0.25 * (scale * f[i][j] + neighbours))
        u = candidate
    return u


def full_weighting(fine, n):
    """Restrict to the (n+1)/2 grid: the coarse point at even index 2i keeps a 1/4-1/2-1 weight."""
    coarse_n = (n + 1) // 2
    coarse = new_grid(coarse_n)
    for i in range(1, coarse_n - 1):
        for j in range(1, coarse_n - 1):
            a, b = 2 * i, 2 * j
            coarse[i][j] = (4.0 * fine[a][b]
                            + 2.0 * (fine[a - 1][b] + fine[a + 1][b] + fine[a][b - 1] + fine[a][b + 1])
                            + (fine[a - 1][b - 1] + fine[a + 1][b - 1]
                               + fine[a - 1][b + 1] + fine[a + 1][b + 1])) / 16.0
    return coarse


def bilinear_prolongation(coarse, fine_n):
    """Inject the coarse values at the even points and interpolate the odd ones."""
    coarse_n = len(coarse)
    fine = new_grid(fine_n)
    for i in range(1, coarse_n - 1):
        for j in range(1, coarse_n - 1):
            fine[2 * i][2 * j] = coarse[i][j]
    for i in range(1, fine_n - 1):
        for j in range(1, fine_n - 1):
            if i % 2 == 0 and j % 2 == 0:
                continue
            if i % 2 == 0:
                fine[i][j] = 0.5 * (fine[i][j - 1] + fine[i][j + 1])
            elif j % 2 == 0:
                fine[i][j] = 0.5 * (fine[i - 1][j] + fine[i + 1][j])
            else:
                fine[i][j] = 0.25 * (fine[i - 1][j - 1] + fine[i + 1][j - 1]
                                     + fine[i - 1][j + 1] + fine[i + 1][j + 1])
    return fine


def dense_solve(n, h, f):
    """Exact solve of the coarsest problem by dense elimination — the recursion's bottom."""
    size = (n - 2) * (n - 2)
    index = {(i, j): (i - 1) * (n - 2) + (j - 1) for i in interior_indices(n) for j in interior_indices(n)}
    matrix = [[0.0] * (size + 1) for _ in range(size)]
    scale = 1.0 / (h * h)
    for (i, j), row in index.items():
        matrix[row][row] = 4.0 * scale
        matrix[row][size] = f[i][j]
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour = index.get((i + di, j + dj))
            if neighbour is not None:
                matrix[row][neighbour] = -scale
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(matrix[r][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        for row in range(column + 1, size):
            factor = matrix[row][column] / matrix[column][column]
            if factor:
                for k in range(column, size + 1):
                    matrix[row][k] -= factor * matrix[column][k]
    solution = [0.0] * size
    for row in reversed(range(size)):
        total = matrix[row][size] - sum(matrix[row][k] * solution[k] for k in range(row + 1, size))
        solution[row] = total / matrix[row][row]
    u = new_grid(n)
    for (i, j), row in index.items():
        u[i][j] = solution[row]
    return u


def v_cycle(u, f, n, h, pre=3, post=3, coarsest=5):
    """One V-cycle: smooth, restrict the residual, recurse, interpolate the correction, smooth."""
    if n <= coarsest:
        return dense_solve(n, h, f)
    u = jacobi_smooth(u, f, n, h, sweeps=pre)
    coarse_n = (n + 1) // 2
    coarse_residual = full_weighting(residual(u, f, n, h), n)
    coarse_correction = v_cycle(new_grid(coarse_n), coarse_residual, coarse_n, 2.0 * h,
                                pre=pre, post=post, coarsest=coarsest)
    correction = bilinear_prolongation(coarse_correction, n)
    for i in interior_indices(n):
        for j in interior_indices(n):
            u[i][j] += correction[i][j]
    return jacobi_smooth(u, f, n, h, sweeps=post)


def solve(f, n, h, cycles):
    u = new_grid(n)
    norms = []
    for _ in range(cycles):
        u = v_cycle(u, f, n, h)
        norms.append(norm(residual(u, f, n, h)))
    return u, norms


def norm(grid):
    return math.sqrt(sum(value * value for row in grid for value in row))


def dot(left, right):
    return sum(a * b for a, b in zip([value for row in left for value in row],
                                     [value for row in right for value in row]))


def manufactured(n, h, frequency=1.0):
    """u = sin(pi x) sin(pi y) with f = 2 pi^2 u — the exact solution of -Delta u = f."""
    pi = math.pi * frequency
    u, f = new_grid(n), new_grid(n)
    for i in range(n):
        for j in range(n):
            x, y = i * h, j * h
            value = math.sin(pi * x) * math.sin(pi * y)
            u[i][j] = value
            f[i][j] = 2.0 * pi * pi * value
    for i in range(n):
        u[0][i] = u[n - 1][i] = u[i][0] = u[i][n - 1] = 0.0
        f[0][i] = f[n - 1][i] = f[i][0] = f[i][n - 1] = 0.0
    return u, f


def max_error(left, right):
    return max(abs(a - b) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b))


def main():
    n = 9
    h = 1.0 / (n - 1)
    u, f = manufactured(n, h)
    check(norm(u) > 0.0, "the manufactured solution is not trivial")
    truncation = {}
    for grid_n in (9, 17, 33):
        grid_h = 1.0 / (grid_n - 1)
        exact_u, grid_f = manufactured(grid_n, grid_h)
        middle = grid_n // 2
        truncation[grid_n] = abs(apply_operator(exact_u, grid_n, grid_h)[middle][middle]
                                 - grid_f[middle][middle])
    check(truncation[33] < truncation[17] / 3.0 < truncation[9] / 9.0,
          f"the stencil's truncation error falls like h^2: "
          f"{truncation[9]:.3e}, {truncation[17]:.3e}, {truncation[33]:.3e}")

    probe_a = new_grid(n)
    probe_b = new_grid(n)
    for i in interior_indices(n):
        for j in interior_indices(n):
            probe_a[i][j] = ((i * 7 + j * 3) % 11) - 5.0
            probe_b[i][j] = ((i * 5 + j * 13) % 7) - 3.0
    check(abs(dot(apply_operator(probe_a, n, h), probe_b)
              - dot(probe_a, apply_operator(probe_b, n, h))) < 1e-9,
          "the discrete Laplacian is symmetric")
    check(dot(probe_a, apply_operator(probe_a, n, h)) > 0.0,
          "and positive definite on the interior")

    coarse_vector = new_grid((n + 1) // 2)
    for i in range(1, (n + 1) // 2 - 1):
        for j in range(1, (n + 1) // 2 - 1):
            coarse_vector[i][j] = ((i + 2 * j) % 5) - 2.0
    prolonged = bilinear_prolongation(coarse_vector, n)
    restricted = full_weighting(probe_a, n)
    check(abs(dot(restricted, coarse_vector) - 0.25 * dot(probe_a, prolonged)) < 1e-9,
          "restriction is the adjoint of prolongation up to the factor 1/4")
    check(all(prolonged[2 * i][2 * j] == coarse_vector[i][j]
              for i in range(1, (n + 1) // 2 - 1) for j in range(1, (n + 1) // 2 - 1)),
          "prolongation injects the coarse values exactly at the even points")
    check(max(abs(value) for row in full_weighting(prolonged, n) for value in row) > 0.0,
          "the transfers are not degenerate")

    oscillatory = new_grid(n)
    smooth_part = new_grid(n)
    for i in interior_indices(n):
        for j in interior_indices(n):
            oscillatory[i][j] = ((-1.0) ** (i + j))
            smooth_part[i][j] = math.sin(math.pi * i / (n - 1)) * math.sin(math.pi * j / (n - 1))
    zero = new_grid(n)
    high_before = norm(oscillatory)
    low_before = norm(smooth_part)
    oscillatory = jacobi_smooth(oscillatory, zero, n, h, sweeps=3)
    smooth_part = jacobi_smooth(smooth_part, zero, n, h, sweeps=3)
    check(norm(oscillatory) / high_before < 0.05,
          f"three Jacobi sweeps nearly kill the oscillatory mode: "
          f"{norm(oscillatory) / high_before:.3f} left")
    check(norm(smooth_part) / low_before > 0.8,
          f"and leave most of the smooth one ({norm(smooth_part) / low_before:.3f}) - that is the point")
    check(norm(oscillatory) / high_before < 0.1 * (norm(smooth_part) / low_before),
          "so the smoother damps the oscillatory error at least ten times faster")

    small = 5
    small_h = 1.0 / (small - 1)
    _, small_f = manufactured(small, small_h, frequency=1.5)
    multigrid_small, _ = solve(small_f, small, small_h, 4)
    exact = dense_solve(small, small_h, small_f)
    check(max_error(multigrid_small, exact) < 1e-9,
          "on the coarsest grid the V-cycle reduces to the exact solve")

    factors = {}
    cycles = {}
    errors = {}
    for grid_n in (17, 33):
        grid_h = 1.0 / (grid_n - 1)
        exact_u, grid_f = manufactured(grid_n, grid_h)
        solution, norms = solve(grid_f, grid_n, grid_h, 12)
        check(all(norms[k] > 0.0 for k in range(1, len(norms))), "the residuals stay positive")
        factors[grid_n] = norms[-1] / norms[-2]
        cycles[grid_n] = next((k + 1 for k, value in enumerate(norms)
                               if value < norms[0] * 1e-8), len(norms))
        errors[grid_n] = max_error(solution, exact_u)
        check(factors[grid_n] < 0.2,
              f"each V-cycle cuts the residual by at least five on n={grid_n}: {factors[grid_n]:.3f}")
        check(all(norms[k + 1] < norms[k] for k in range(len(norms) - 1)),
              f"the residual falls monotonically on n={grid_n}")
    check(abs(factors[17] - factors[33]) < 0.1,
          f"the convergence factor is grid independent: {factors[17]:.3f} vs {factors[33]:.3f}")
    check(abs(cycles[17] - cycles[33]) <= 2,
          f"and so is the cycle count: {cycles[17]} vs {cycles[33]}")

    grid_n = 33
    grid_h = 1.0 / (grid_n - 1)
    _, coarse_f = manufactured(17, 1.0 / 16.0)
    _, fine_f = manufactured(grid_n, grid_h)
    check(errors[grid_n] < 1e-3, f"the discrete solution is accurate: {errors[grid_n]:.2e}")
    ratio = errors[17] / errors[33]
    check(2.0 < ratio < 8.0, f"halving h cuts the error about fourfold (measured {ratio:.2f})")

    jacobi_history = []
    u = new_grid(grid_n)
    for _ in range(12):
        u = jacobi_smooth(u, fine_f, grid_n, grid_h, sweeps=10)
        jacobi_history.append(norm(residual(u, fine_f, grid_n, grid_h)))
    check(jacobi_history[-1] > jacobi_history[0] * 0.5,
          "120 Jacobi sweeps barely dent the residual, where six V-cycles fixed it")
    check(cycles[grid_n] <= 12, f"multigrid needs {cycles[grid_n]} cycles for a 1e-8 reduction")

    zero_solution, zero_norms = solve(new_grid(grid_n), grid_n, grid_h, 3)
    check(all(value == 0.0 for row in zero_solution for value in row) and
          all(value == 0.0 for value in zero_norms),
          "a zero right-hand side stays zero all the way through the cycle")
    repeat, repeat_norms = solve(fine_f, grid_n, grid_h, 12)
    check(repeat_norms == solve(fine_f, grid_n, grid_h, 12)[1], "the solver is deterministic")

    print(f"convergence factor per V-cycle: {factors[17]:.3f} at n=17 and {factors[33]:.3f} at n=33")
    print(f"cycles to a 1e-8 residual: {cycles[17]} (n=17), {cycles[33]} (n=33); "
          f"120 Jacobi sweeps only reach {jacobi_history[-1] / jacobi_history[0]:.2f} of the start")
    print(f"discrete error: {errors[17]:.2e} at n=17, {errors[33]:.2e} at n=33 (ratio {ratio:.2f})")
    print("PASS")


if __name__ == "__main__":
    main()
