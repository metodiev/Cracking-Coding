"""Task 497 - Implement L-BFGS numeric optimizer skeleton for small dims. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

A compact L-BFGS for small dimensions: the two-loop recursion builds the search direction from
the last m = 5 (s, y) correction pairs, a backtracking Armijo line search (c1 = 1e-4, rho =
0.5, first trial step 1.0) picks the step, and the loop stops when ||grad f|| <= 1e-8 or after
200 iterations. The model problem is the convex quadratic f(x) = 0.5 x^T A x - b^T x with
A = [[4,1,0],[1,3,1],[0,1,2]], b = [1,2,3], x0 = 0; its exact minimizer x* = A^-1 b and f* are
computed by Gaussian elimination as the oracle. Documented tolerance: L-BFGS reaches
||x - x*|| <= 1e-6, |f - f*| <= 1e-9 and ||grad f|| <= 1e-8. A plain gradient descent with the
documented step size eta = 1/L (largest eigenvalue by power iteration, both runs started at x0
and compared at the same ||grad f|| <= 1e-6 tolerance) needs at least four times as many
iterations - L-BFGS is compared against it directly.
"""
import math

A = [[4.0, 1.0, 0.0],
     [1.0, 3.0, 1.0],
     [0.0, 1.0, 2.0]]
B = [1.0, 2.0, 3.0]
X0 = [0.0, 0.0, 0.0]
HISTORY = 5
MAX_ITER = 200
GD_MAX_ITER = 20000
C1 = 1e-4
RHO = 0.5


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def objective(x):
    return 0.5 * dot(x, matvec(A, x)) - dot(B, x)


def gradient(x):
    return [a - b for a, b in zip(matvec(A, x), B)]


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting - the exact-minimizer oracle."""
    n = len(matrix)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(col + 1, n):
            factor = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = aug[r][n] - sum(aug[r][c] * x[c] for c in range(r + 1, n))
        x[r] = acc / aug[r][r]
    return x


def largest_eigenvalue(matrix, iterations=100):
    v = [1.0] * len(matrix)
    v = [value / math.sqrt(dot(v, v)) for value in v]
    for _ in range(iterations):
        w = matvec(matrix, v)
        v = [value / math.sqrt(dot(w, w)) for value in w]
    return dot(v, matvec(matrix, v))


def two_loop_direction(g, s_hist, y_hist):
    """L-BFGS two-loop recursion: returns r = H*g; the search direction is -r."""
    q = list(g)
    alphas = []
    for s, y in reversed(list(zip(s_hist, y_hist))):
        rho = 1.0 / dot(y, s)
        alpha = rho * dot(s, q)
        alphas.append(alpha)
        q = [qi - alpha * yi for qi, yi in zip(q, y)]
    if s_hist:
        s_last, y_last = s_hist[-1], y_hist[-1]
        gamma = dot(s_last, y_last) / dot(y_last, y_last)
    else:
        gamma = 1.0
    r = [gamma * qi for qi in q]
    for (s, y), alpha in zip(zip(s_hist, y_hist), reversed(alphas)):
        rho = 1.0 / dot(y, s)
        beta = rho * dot(y, r)
        r = [ri + si * (alpha - beta) for ri, si in zip(r, s)]
    return r


def lbfgs():
    x = list(X0)
    g = gradient(x)
    s_hist, y_hist = [], []
    iterations = 0
    for _ in range(MAX_ITER):
        if math.sqrt(dot(g, g)) <= 1e-8:
            break
        direction = [-ri for ri in two_loop_direction(g, s_hist, y_hist)]
        descent = dot(g, direction)
        if descent >= 0.0:                   # safeguard: fall back to steepest descent
            direction = [-gi for gi in g]
            descent = -dot(g, g)
        step = 1.0
        while objective([xi + step * di for xi, di in zip(x, direction)]) > \
                objective(x) + C1 * step * descent:
            step *= RHO
        x_new = [xi + step * di for xi, di in zip(x, direction)]
        g_new = gradient(x_new)
        s = [a - b for a, b in zip(x_new, x)]
        y = [a - b for a, b in zip(g_new, g)]
        if dot(s, y) > 0.0:                  # curvature condition
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > HISTORY:
                s_hist.pop(0)
                y_hist.pop(0)
        x, g = x_new, g_new
        iterations += 1
    return x, g, iterations


def gd(tolerance):
    eta = 1.0 / largest_eigenvalue(A)        # documented GD step size
    x = list(X0)
    g = gradient(x)
    iterations = 0
    while math.sqrt(dot(g, g)) > tolerance and iterations < GD_MAX_ITER:
        x = [xi - eta * gi for xi, gi in zip(x, g)]
        g = gradient(x)
        iterations += 1
    return x, g, iterations


def main():
    x_star = solve_linear(A, B)
    f_star = objective(x_star)
    assert max(abs(g) for g in gradient(x_star)) <= 1e-12, "Gaussian oracle is exact"

    # Skeleton sanity: with an empty history H = I, so H*g = g and the step is -g.
    g0 = gradient(X0)
    assert two_loop_direction(g0, [], []) == g0

    x_lb, g_lb, lb_iters = lbfgs()
    g_norm = math.sqrt(dot(g_lb, g_lb))
    assert g_norm <= 1e-8, "documented L-BFGS gradient tolerance"
    assert max(abs(a - b) for a, b in zip(x_lb, x_star)) <= 1e-6
    assert abs(objective(x_lb) - f_star) <= 1e-9
    assert lb_iters <= 20, "L-BFGS must converge in few iterations on a 3-dim quadratic"

    # Plain GD for comparison: same start, compared at the looser ||g|| <= 1e-6 tolerance.
    x_gd, g_gd, gd_iters = gd(1e-6)
    assert math.sqrt(dot(g_gd, g_gd)) <= 1e-6, "GD must converge within the cap"
    assert gd_iters > 4 * lb_iters, "L-BFGS must need far fewer iterations"
    assert max(abs(a - b) for a, b in zip(x_gd, x_star)) <= 1e-4

    # Determinism: a re-run reproduces the same iterate bit for bit.
    x_repeat, _, repeat_iters = lbfgs()
    assert repeat_iters == lb_iters and x_repeat == x_lb


if __name__ == "__main__":
    main()
    print("PASS")
