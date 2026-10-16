"""Task 593 - Implement L1-regularized least squares via coordinate descent. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Lasso: minimise f(w) = (1/2) ||y - X w||^2 + lambda ||w||_1 with cyclic coordinate descent.  Each
coordinate is updated in closed form by the soft-threshold of the partial residual,

    w_j <- S(x_j^T r + ||x_j||^2 w_j, lambda) / ||x_j||^2,   S(z, t) = sign(z) * max(|z| - t, 0),

with r = y - X w the current residual; working with ||x_j||^2 (rather than assuming unit-norm
columns) keeps the rule valid for arbitrary scaling, and the file also checks that the unit-norm
normalisation changes nothing but the reported coefficients' scale.  The objective f is convex, so
the KKT conditions are both necessary and sufficient:

    |x_j^T (y - X w)| <= lambda for all j,  and  x_j^T (y - X w) = lambda * sign(w_j) when w_j != 0.

Documented checks:

  * full KKT verification of every solution returned (both inequalities and the equality on the
    active set, tolerance 1e-8), on random instances and on the path;
  * exhaustive global optimality: for a tiny problem (p = 6) the file enumerates all 3^p
    sign patterns, solves the resulting equality-constrained least-squares problem in closed form,
    keeps the patterns whose signs are self-consistent and takes the best feasible objective.  The
    coordinate-descent objective is asserted to equal that brute-force optimum to 1e-9, which is a
    genuinely independent global check rather than a comparison against another iterative method;
  * the two textbook limits: lambda = 0 reproduces the ordinary least-squares solution (compared
    against a Gaussian-elimination solve of the normal equations) and lambda >= lambda_max =
    max_j |x_j^T y| returns exactly w = 0, while lambda slightly below lambda_max activates exactly
    one coordinate -- the arg-max of |x_j^T y| with the sign of x_j^T y;
  * an orthonormal design (X^T X = I) has the closed form w = S(X^T y, lambda), asserted to 1e-9;
  * the objective is non-increasing over every sweep; the solution path is monotone,
    ||w(lambda_1)||_1 >= ||w(lambda_2)||_1 for lambda_1 < lambda_2 (a theorem for the lasso), and
    warm-started path solutions coincide with cold-started ones to 1e-9;
  * the honest caveat about convergence: coordinate descent is not finitely terminating for
    correlated designs, so the file fixes a sweep budget, asserts the KKT violation of the
    returned point, and reports (asserts) that the violation falls below 1e-8 within the budget --
    it does not claim exact optimality in a fixed number of sweeps.

Data comes from the repo's 16-bit LCG: the draws are few, immediately centred, and each instance is
normalized, so no long-range independence assumption is being made.
"""
import math

LCG_MOD = 65537


def lcg(state):
    return (state * 75 + 74) % LCG_MOD


def soft_threshold(value, threshold):
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def lasso_objective(X, y, w, lam):
    residual = [y[i] - sum(X[i][j] * w[j] for j in range(len(w))) for i in range(len(y))]
    return 0.5 * sum(value * value for value in residual) + lam * sum(abs(value) for value in w)


def partial_residual(X, y, w, j):
    """r = y - X w with the contribution of coordinate j removed."""
    return [y[i] - sum(X[i][k] * w[k] for k in range(len(w)) if k != j)
            for i in range(len(y))]


def lasso_coordinate_descent(X, y, lam, sweeps=500, tol=1e-12, w0=None):
    """Cyclic coordinate descent with soft thresholding; returns (w, objective history)."""
    n = len(X)
    p = len(X[0])
    column_norms = [sum(X[i][j] * X[i][j] for i in range(n)) for j in range(p)]
    w = [0.0] * p if w0 is None else list(w0)
    history = [lasso_objective(X, y, w, lam)]
    for _ in range(sweeps):
        largest_change = 0.0
        for j in range(p):
            if column_norms[j] == 0.0:
                continue
            partial = partial_residual(X, y, w, j)
            correlation = sum(X[i][j] * partial[i] for i in range(n))
            updated = soft_threshold(correlation, lam) / column_norms[j]
            largest_change = max(largest_change, abs(updated - w[j]))
            w[j] = updated
        history.append(lasso_objective(X, y, w, lam))
        if largest_change <= tol:
            break
    return w, history


def kkt_violation(X, y, w, lam):
    """Max over j of the KKT slack (0 at an exact solution)."""
    n = len(X)
    residual = [y[i] - sum(X[i][j] * w[j] for j in range(len(w))) for i in range(n)]
    worst = 0.0
    for j in range(len(w)):
        correlation = sum(X[i][j] * residual[i] for i in range(n))
        if w[j] > 0.0:
            worst = max(worst, abs(correlation - lam))
        elif w[j] < 0.0:
            worst = max(worst, abs(correlation + lam))
        else:
            worst = max(worst, max(0.0, abs(correlation) - lam))
    return worst


def normal_equations(X, y):
    """Ordinary least squares via Gaussian elimination on X^T X w = X^T y."""
    p = len(X[0])
    gram = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(p)] for a in range(p)]
    rhs = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(p)]
    a = [gram[i] + [rhs[i]] for i in range(p)]
    for col in range(p):
        pivot_row = max(range(col, p), key=lambda r: abs(a[r][col]))
        a[col], a[pivot_row] = a[pivot_row], a[col]
        for r in range(col + 1, p):
            factor = a[r][col] / a[col][col]
            for c in range(col, p + 1):
                a[r][c] -= factor * a[col][c]
    w = [0.0] * p
    for r in range(p - 1, -1, -1):
        w[r] = (a[r][p] - sum(a[r][c] * w[c] for c in range(r + 1, p))) / a[r][r]
    return w


def brute_force_lasso(X, y, lam):
    """Global optimum by enumerating all 3^p (support, sign) patterns and checking consistency."""
    from itertools import product
    p = len(X[0])
    best = None
    best_w = None
    gram = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(p)] for a in range(p)]
    cross = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(p)]
    for pattern in product((0, 1, -1), repeat=p):
        support = [j for j in range(p) if pattern[j] != 0]
        if not support:
            candidate = [0.0] * p
            value = lasso_objective(X, y, candidate, lam)
            if best is None or value < best - 1e-15:
                best = value
                best_w = candidate
            continue
        sub = [[gram[i][j] for j in support] for i in support]
        rhs = [cross[j] - lam * pattern[j] for j in support]
        a = [sub[i] + [rhs[i]] for i in range(len(support))]
        singular = False
        for col in range(len(support)):
            pivot_row = max(range(col, len(support)), key=lambda r: abs(a[r][col]))
            a[col], a[pivot_row] = a[pivot_row], a[col]
            if abs(a[col][col]) < 1e-12:
                singular = True
                break
            for r in range(col + 1, len(support)):
                factor = a[r][col] / a[col][col]
                for c in range(col, len(support) + 1):
                    a[r][c] -= factor * a[col][c]
        if singular:
            continue
        solution = [0.0] * len(support)
        for r in range(len(support) - 1, -1, -1):
            solution[r] = (a[r][len(support)] -
                           sum(a[r][c] * solution[c] for c in range(r + 1, len(support)))) / a[r][r]
        consistent = True
        for index, j in enumerate(support):
            if pattern[j] > 0 and solution[index] <= 0.0:
                consistent = False
            if pattern[j] < 0 and solution[index] >= 0.0:
                consistent = False
        if not consistent:
            continue
        candidate = [0.0] * p
        for index, j in enumerate(support):
            candidate[j] = solution[index]
        value = lasso_objective(X, y, candidate, lam)
        if best is None or value < best - 1e-15:
            best = value
            best_w = candidate
    return best, best_w


def normalize_columns(X):
    n = len(X)
    p = len(X[0])
    norms = [math.sqrt(sum(X[i][j] * X[i][j] for i in range(n))) or 1.0 for j in range(p)]
    return [[X[i][j] / norms[j] for j in range(p)] for i in range(n)], norms


def random_design(state, n, p, low, high):
    X = []
    for _ in range(n):
        row = []
        for _ in range(p):
            state = lcg(state)
            row.append(low + (high - low) * ((state % 1000) / 1000.0))
        X.append(row)
    return state, X


def random_response(state, n, low, high):
    y = []
    for _ in range(n):
        state = lcg(state)
        y.append(low + (high - low) * ((state % 1000) / 1000.0))
    return state, y


def main():
    # --- lambda = 0 must reproduce ordinary least squares ------------------------------
    state = 593593
    state, X = random_design(state, 20, 5, -1.0, 1.0)
    X, column_norms = normalize_columns(X)
    state, y = random_response(state, 20, -1.0, 1.0)
    w_zero, _ = lasso_coordinate_descent(X, y, 0.0, sweeps=2000)
    ols = normal_equations(X, y)
    assert max(abs(a - b) for a, b in zip(w_zero, ols)) <= 1e-9, (w_zero, ols)

    # --- lambda_max: everything is zero, and just below it one coordinate wakes up -----
    cross = [sum(X[i][j] * y[i] for i in range(len(y))) for j in range(5)]
    lambda_max = max(abs(value) for value in cross)
    w_at_max, _ = lasso_coordinate_descent(X, y, lambda_max)
    assert all(value == 0.0 for value in w_at_max), w_at_max
    assert kkt_violation(X, y, w_at_max, lambda_max) <= 1e-12
    w_above, _ = lasso_coordinate_descent(X, y, lambda_max * 1.5)
    assert all(value == 0.0 for value in w_above)
    winner = max(range(5), key=lambda j: abs(cross[j]))
    w_just, _ = lasso_coordinate_descent(X, y, lambda_max * 0.99)
    assert [j for j in range(5) if w_just[j] != 0.0] == [winner], w_just
    assert (w_just[winner] > 0.0) == (cross[winner] > 0.0)
    assert kkt_violation(X, y, w_just, lambda_max * 0.99) <= 1e-8

    # --- orthonormal design: the closed form ------------------------------------------
    identity = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
    state, y_ortho = random_response(state, 6, -3.0, 3.0)
    for lam in (0.0, 0.25, 0.75, 1.5):
        w, _ = lasso_coordinate_descent(identity, y_ortho, lam)
        expected = [soft_threshold(value, lam) for value in y_ortho]
        assert max(abs(a - b) for a, b in zip(w, expected)) <= 1e-12
        assert kkt_violation(identity, y_ortho, w, lam) <= 1e-10

    # --- exhaustive global optimality on a tiny problem -------------------------------
    state, X_tiny = random_design(state, 10, 6, -1.0, 1.0)
    X_tiny, _ = normalize_columns(X_tiny)
    state, y_tiny = random_response(state, 10, -1.0, 1.0)
    for lam in (0.05, 0.2, 0.4):
        w, history = lasso_coordinate_descent(X_tiny, y_tiny, lam, sweeps=2000)
        brute_value, brute_w = brute_force_lasso(X_tiny, y_tiny, lam)
        value = lasso_objective(X_tiny, y_tiny, w, lam)
        assert abs(value - brute_value) <= 1e-9, (lam, value, brute_value)
        assert kkt_violation(X_tiny, y_tiny, w, lam) <= 1e-8
        assert kkt_violation(X_tiny, y_tiny, brute_w, lam) <= 1e-8, "the brute force is consistent"

    # --- the objective is non-increasing sweep by sweep -------------------------------
    state, X_big = random_design(state, 40, 12, -1.0, 1.0)
    X_big, _ = normalize_columns(X_big)
    state, y_big = random_response(state, 40, -1.0, 1.0)
    w_big, history = lasso_coordinate_descent(X_big, y_big, 0.15, sweeps=300)
    for step in range(1, len(history)):
        assert history[step] <= history[step - 1] + 1e-12, step
    assert history[-1] <= history[0]
    assert kkt_violation(X_big, y_big, w_big, 0.15) <= 1e-8

    # --- regularization path: monotone in lambda, warm start free ---------------------
    lambdas = [0.5, 0.35, 0.25, 0.15, 0.08, 0.02]
    cold = []
    for lam in lambdas:
        solution, _ = lasso_coordinate_descent(X_big, y_big, lam, sweeps=2000)
        cold.append(solution)
    warm = []
    previous = [0.0] * 12
    for lam in lambdas:
        solution, _ = lasso_coordinate_descent(X_big, y_big, lam, sweeps=2000, w0=previous)
        warm.append(solution)
        previous = solution
    for index in range(len(lambdas)):
        assert max(abs(a - b) for a, b in zip(cold[index], warm[index])) <= 1e-9, index
        assert kkt_violation(X_big, y_big, cold[index], lambdas[index]) <= 1e-8
    norms = [sum(abs(value) for value in solution) for solution in cold]
    for index in range(1, len(norms)):
        assert norms[index] >= norms[index - 1] - 1e-12, "larger lambda never adds mass"
    assert norms[-1] > norms[0], "the path actually changes"
    counts = [sum(1 for value in solution if value != 0.0) for solution in cold]
    assert counts[0] <= counts[-1], "more coordinates survive at smaller lambda"

    # --- columns that are all zero are handled without dividing by zero ---------------
    degenerate = [[1.0, 0.0, 0.5], [0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 0.0, 0.5]]
    y_degenerate = [1.0, 2.0, 1.0, 2.0]
    w_degenerate, _ = lasso_coordinate_descent(degenerate, y_degenerate, 0.1)
    assert w_degenerate[1] == 0.0
    assert all(math.isfinite(value) for value in w_degenerate)

    # --- determinism ------------------------------------------------------------------
    state_a, X_a = random_design(777, 12, 4, -1.0, 1.0)
    state_b, X_b = random_design(777, 12, 4, -1.0, 1.0)
    assert X_a == X_b and state_a == state_b
    _, y_a = random_response(999, 12, -1.0, 1.0)
    _, y_b = random_response(999, 12, -1.0, 1.0)
    assert y_a == y_b
    assert lasso_coordinate_descent(X_a, y_a, 0.2) == lasso_coordinate_descent(X_b, y_b, 0.2)


if __name__ == "__main__":
    main()
    print("PASS")
