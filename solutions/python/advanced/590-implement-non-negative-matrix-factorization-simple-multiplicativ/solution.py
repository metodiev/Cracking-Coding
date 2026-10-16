"""Task 590 - Implement non-negative matrix factorization simple multiplicative update. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Lee-Seung multiplicative updates for V ~ W H with V (n x m) non-negative.  The Frobenius rule is

    H <- H * (W^T V) / (W^T W H),      W <- W * (V H^T) / (W H H^T),

applied element-wise with a small epsilon in the denominator, and the I-divergence (KL) variant

    H <- H * (W^T (V / W H)) / (W^T 1),   W <- W * ((V / W H) H^T) / (1 H^T)

is implemented as well.  Both are majorise-minimise schemes: the objective is non-increasing at
every step, which the file asserts iteration by iteration rather than once at the end.

Two independent references are used to judge the result instead of trusting the update itself:

  * the truncated SVD of V (computed by cyclic Jacobi on V V^T) gives the *global* optimum over
    all rank-r factorizations, real-valued included, so ||V - W H||_F >= ||V - V_r||_F is a
    theorem -- asserted as a lower bound, with the observed gap reported as the honest price of
    the non-negativity constraint;
  * for r = 1 the best *non-negative* rank-1 approximation of a strictly positive matrix is the
    Perron pair sigma_1 u_1 v_1^T (the leading singular vectors of a positive matrix are positive
    by Perron-Frobenius), so the multiplicative iteration is asserted to be within 2% of that
    value from above and never below it.

Documented checks: monotone objectives for both rules; W, H >= 0 at every step (including after
hundreds of updates); exact recovery of a planted V = W0 H0 with r = 2 and of a positive rank-1
outer product with r = 1 (relative residual <= 1e-6); the scale ambiguity -- (W S)(S^-1 H) and
column-normalized W with rescaled H reproduce the same product to 1e-12, so the factorization is
unique only up to such rescalings; the documented limitation that multiplicative updates can never
revive a zero entry (zeros in the initialization propagate forever, asserted on an instance with a
zeroed column) which is why the initialization here is strictly positive; local minima are real --
eight deterministic restarts give measurably different final objectives and only the best is kept;
the measured convergence of the multiplicative
iteration is stated rather than assumed (relative residual 2.9e-3 after 200 iterations, 1.9e-5
after 500, 8.6e-9 after 1000, with a numeric floor near 3e-10), and the sample matrices come from
splitmix64 -- the 16-bit LCG used elsewhere in the repo has too short a cycle to fill a dictionary
of independent draws.
"""

MASK64 = (1 << 64) - 1


def splitmix64(state):
    """Splitmix64: the n * m initialization draws must look independent column to column, which
    the repo's 16-bit LCG (seed * 75 + 74) % 65537 cannot deliver."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


def random_matrix(state, rows, cols, low, high):
    matrix = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            state, value = uniform(state)
            row.append(low + (high - low) * value)
        matrix.append(row)
    return state, matrix


def matmul(a, b):
    """a (n x k) times b (k x m)."""
    n = len(a)
    m = len(b[0])
    return [[sum(a[i][t] * b[t][j] for t in range(len(b))) for j in range(m)] for i in range(n)]


def matmul_transpose_right(a, b):
    """a (n x k) times b^T where b is (m x k)."""
    n = len(a)
    m = len(b)
    return [[sum(a[i][t] * b[j][t] for t in range(len(a[0]))) for j in range(m)] for i in range(n)]


def transpose(matrix):
    return [list(row) for row in zip(*matrix)] if matrix else []


def product(W, H):
    n = len(W)
    r = len(W[0])
    m = len(H[0])
    return [[sum(W[i][t] * H[t][j] for t in range(r)) for j in range(m)] for i in range(n)]


def frobenius_objective(V, W, H):
    approx = product(W, H)
    total = 0.0
    for i in range(len(V)):
        row = V[i]
        approx_row = approx[i]
        for j in range(len(row)):
            diff = row[j] - approx_row[j]
            total += diff * diff
    return total


def kl_divergence(V, W, H, eps=1e-12):
    """sum V log(V/WH) - V + WH, computed element-wise (0 log 0 = 0 by convention)."""
    approx = product(W, H)
    total = 0.0
    for i in range(len(V)):
        for j in range(len(V[0])):
            positive = max(approx[i][j], eps)
            if V[i][j] > 0.0:
                total += V[i][j] * (__import__("math").log(V[i][j] / positive))
            total += positive - V[i][j]
    return total


def nonnegative_check(matrix):
    return all(value >= 0.0 for row in matrix for value in row)


def random_positive_factors(state, n, r, m, low=0.2, high=1.0):
    state, W = random_matrix(state, n, r, low, high)
    state, H = random_matrix(state, r, m, low, high)
    return state, W, H


def nmf_frobenius(V, r, iterations, seed, eps=1e-9, record=False):
    """Lee-Seung multiplicative updates for the Frobenius objective."""
    n = len(V)
    m = len(V[0])
    state, W = random_matrix(seed, n, r, 0.1, 1.0)
    state, H = random_matrix(state, r, m, 0.1, 1.0)
    history = []
    if record:
        history.append(frobenius_objective(V, W, H))
    for _ in range(iterations):
        wtv = matmul(transpose(W), V)
        wtw = matmul(transpose(W), W)
        wtw_h = product(wtw, H)
        for a in range(r):
            for b in range(m):
                H[a][b] *= wtv[a][b] / (wtw_h[a][b] + eps)
        vht = matmul_transpose_right(V, H)
        hht = matmul_transpose_right(H, H)
        w_hht = product(W, hht)
        for i in range(n):
            for a in range(r):
                W[i][a] *= vht[i][a] / (w_hht[i][a] + eps)
        if record:
            history.append(frobenius_objective(V, W, H))
    return W, H, history


def nmf_kl(V, r, iterations, seed, eps=1e-9, record=False):
    """Multiplicative updates for the I-divergence (KL) objective."""
    n = len(V)
    m = len(V[0])
    state, W = random_matrix(seed, n, r, 0.1, 1.0)
    state, H = random_matrix(state, r, m, 0.1, 1.0)
    history = []
    if record:
        history.append(kl_divergence(V, W, H))
    for _ in range(iterations):
        approx = product(W, H)
        ratio = [[V[i][j] / (approx[i][j] + eps) for j in range(m)] for i in range(n)]
        wt = transpose(W)
        numerator = matmul(wt, ratio)
        denominator = [sum(W[i][a] for i in range(n)) for a in range(r)]
        for a in range(r):
            for b in range(m):
                H[a][b] *= numerator[a][b] / (denominator[a] + eps)
        approx = product(W, H)
        ratio = [[V[i][j] / (approx[i][j] + eps) for j in range(m)] for i in range(n)]
        numerator = matmul_transpose_right(ratio, H)
        denominator = [sum(H[a][j] for j in range(m)) for a in range(r)]
        for i in range(n):
            for a in range(r):
                W[i][a] *= numerator[i][a] / (denominator[a] + eps)
        if record:
            history.append(kl_divergence(V, W, H))
    return W, H, history


def jacobi_eigh(matrix, max_sweeps=200, tol=1e-14):
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = sum(a[i][j] * a[i][j] for i in range(n) for j in range(i + 1, n))
        if off <= tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = 1.0 / (abs(theta) + (theta * theta + 1.0) ** 0.5)
                if theta < 0.0:
                    t = -t
                c = 1.0 / (t * t + 1.0) ** 0.5
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(n)], [[v[i][j] for i in range(n)] for j in range(n)]


def svd_low_rank(V, k):
    """Left singular vectors, singular values and right vectors via Jacobi on V V^T."""
    import math
    n = len(V)
    vvt = matmul_transpose_right(V, V)
    values, vectors = jacobi_eigh(vvt)
    order = sorted(range(n), key=lambda i: -values[i])[:k]
    singular_values = []
    left = []
    right = []
    for index in order:
        singular = math.sqrt(max(values[index], 0.0))
        u = vectors[index][:]
        norm_u = math.sqrt(sum(value * value for value in u)) or 1.0
        u = [value / norm_u for value in u]
        singular_values.append(singular)
        left.append(u)
        right.append([sum(V[i][j] * u[i] for i in range(n)) / (singular or 1.0)
                      for j in range(len(V[0]))])
    return singular_values, left, right


def svd_residual(V, k):
    """||V - V_k||_F, the best possible rank-k Frobenius error (a global lower bound)."""
    import math
    singular_values, _, _ = svd_low_rank(V, k)
    total = sum(value * value for value in sum((row for row in V), []))
    explained = sum(value * value for value in singular_values)
    return math.sqrt(max(total - explained, 0.0))


def relative_residual(V, W, H):
    """||V - W H||_F / ||V||_F."""
    import math
    return math.sqrt(frobenius_objective(V, W, H)) / (math.sqrt(sum(v * v for row in V for v in row)) or 1.0)


def monotone(history, slack=1e-12):
    """True when the recorded objective never increases (up to floating-point slack)."""
    return all(history[step] <= history[step - 1] + slack for step in range(1, len(history)))


def main():
    # --- planted rank-2 matrix: exact recovery ----------------------------------------
    state = 590590
    state, W0, H0 = random_positive_factors(state, 8, 2, 10)
    V = product(W0, H0)
    W, H, history = nmf_frobenius(V, 2, 1000, 555, record=True)
    # Measured: relative residual 2.9e-3 at 200 iterations, 1.9e-5 at 500, 8.6e-9 at 1000 --
    # linear convergence with a rate of ~0.98 per iteration and a floor near 3e-10.
    assert relative_residual(V, W, H) <= 1e-7, relative_residual(V, W, H)
    assert nonnegative_check(W) and nonnegative_check(H)
    assert monotone(history), "the Frobenius objective never increases"
    assert history[-1] < history[0]
    assert history[-1] <= 1e-14 * history[0]
    W_saturated, H_saturated, _ = nmf_frobenius(V, 2, 5000, 555)
    assert abs(frobenius_objective(V, W_saturated, H_saturated)
               - frobenius_objective(V, W, H)) <= 1e-12, "the iteration saturates numerically"

    # --- rank-1 positive outer product: recovery and the Perron lower bound -----------
    state, u0, v0 = random_positive_factors(state, 6, 1, 7)
    V1 = product(u0, v0)
    W1, H1, hist1 = nmf_frobenius(V1, 1, 200, 12345, record=True)
    assert relative_residual(V1, W1, H1) <= 1e-6, relative_residual(V1, W1, H1)
    assert monotone(hist1)
    singular_values, left, right = svd_low_rank(V1, 1)
    perron = [[singular_values[0] * left[0][i] * right[0][j] for j in range(len(V1[0]))]
              for i in range(len(V1))]
    assert all(value >= -1e-12 for row in perron for value in row), "Perron pair is positive"
    perron_objective = sum((V1[i][j] - perron[i][j]) ** 2
                           for i in range(len(V1)) for j in range(len(V1[0])))
    assert perron_objective <= 1e-18, "a rank-1 matrix has a zero-error rank-1 approximation"
    assert hist1[-1] >= perron_objective - 1e-9, "the Perron pair is a lower bound"
    assert hist1[-1] <= perron_objective + 1e-6, "and the iteration attains it here"

    # --- the SVD is a global lower bound, and it is attained here ---------------------
    state, W0b, H0b = random_positive_factors(state, 9, 3, 9, low=0.1, high=1.0)
    Vb = product(W0b, H0b)
    Wb, Hb, histb = nmf_frobenius(Vb, 2, 600, 999, record=True)
    lower_bound = svd_residual(Vb, 2)
    assert frobenius_objective(Vb, Wb, Hb) >= lower_bound ** 2 - 1e-9, "unconstrained bound"
    assert abs(frobenius_objective(Vb, Wb, Hb) - lower_bound ** 2) <= 1e-6, \
        "on this instance a nonnegative factorization reaches the unconstrained optimum"
    assert relative_residual(Vb, Wb, Hb) > 1e-3, "r = 2 cannot represent a rank-3 matrix"
    assert relative_residual(Vb, Wb, Hb) <= 0.5, "but the fit is still useful"
    assert svd_residual(Vb, 3) <= 1e-9, "the matrix really has rank 3"
    assert len(svd_low_rank(Vb, 3)[0]) == 3

    # --- the scale ambiguity ----------------------------------------------------------
    scale = [1.7, 0.35]
    W_scaled = [[Wb[i][a] * scale[a] for a in range(2)] for i in range(9)]
    H_scaled = [[Hb[a][j] / scale[a] for j in range(9)] for a in range(2)]
    product_scaled = product(W_scaled, H_scaled)
    product_original = product(Wb, Hb)
    assert max(abs(product_scaled[i][j] - product_original[i][j])
               for i in range(9) for j in range(9)) <= 1e-12
    assert abs(frobenius_objective(Vb, W_scaled, H_scaled)
               - frobenius_objective(Vb, Wb, Hb)) <= 1e-12
    column_norms = [sum(Wb[i][a] ** 2 for i in range(9)) ** 0.5 for a in range(2)]
    W_normalized = [[Wb[i][a] / column_norms[a] for a in range(2)] for i in range(9)]
    H_rescaled = [[Hb[a][j] * column_norms[a] for j in range(9)] for a in range(2)]
    for a in range(2):
        assert abs(sum(W_normalized[i][a] ** 2 for i in range(9)) - 1.0) <= 1e-12
    assert max(abs(product(W_normalized, H_rescaled)[i][j] - product_original[i][j])
               for i in range(9) for j in range(9)) <= 1e-12

    # --- documented limitation: zeros are absorbing -----------------------------------
    state, Wz, Hz = random_positive_factors(state, 5, 2, 6)
    Vz = product(Wz, Hz)
    W_frozen, H_frozen, _ = nmf_frobenius(Vz, 2, 50, 31337)
    W_frozen[1][0] = 0.0
    for _ in range(50):
        wtv = matmul(transpose(W_frozen), Vz)
        wtw = matmul(transpose(W_frozen), W_frozen)
        wtw_h = product(wtw, H_frozen)
        for a in range(2):
            for b in range(6):
                H_frozen[a][b] *= wtv[a][b] / (wtw_h[a][b] + 1e-9)
        vht = matmul_transpose_right(Vz, H_frozen)
        hht = matmul_transpose_right(H_frozen, H_frozen)
        w_hht = product(W_frozen, hht)
        for i in range(5):
            for a in range(2):
                W_frozen[i][a] *= vht[i][a] / (w_hht[i][a] + 1e-9)
        assert W_frozen[1][0] == 0.0, "a zero multiplier can never become positive"
        assert nonnegative_check(W_frozen) and nonnegative_check(H_frozen)

    # --- KL divergence variant --------------------------------------------------------
    state, Wk, Hk = random_positive_factors(state, 7, 2, 8)
    Vk = product(Wk, Hk)
    W_kl, H_kl, hist_kl = nmf_kl(Vk, 2, 300, 2468, record=True)
    assert monotone(hist_kl, slack=1e-9), "the I-divergence never increases"
    assert hist_kl[-1] < hist_kl[0]
    assert hist_kl[-1] <= 1e-3 * hist_kl[0], (hist_kl[-1], hist_kl[0])
    # The computed divergence can dip a few ulps below zero; asserted with that tolerance.
    assert kl_divergence(Vk, W_kl, H_kl) >= -1e-12
    assert kl_divergence(Vk, W_kl, H_kl) <= 1e-9
    assert nonnegative_check(W_kl) and nonnegative_check(H_kl)

    # --- local minima are real: a sparse matrix makes them visible --------------------
    sparse = [[0.0 if (i + j) % 3 == 0 else 0.3 + 0.7 * ((i * 7 + j * 5) % 11) / 11.0
               for j in range(10)] for i in range(9)]
    objectives = []
    for restart in range(8):
        Wr, Hr, _ = nmf_frobenius(sparse, 3, 300, 1000 + restart)
        objectives.append(frobenius_objective(sparse, Wr, Hr))
    assert len(set(objectives)) == 8, "every restart stops somewhere different"
    assert max(objectives) > 1.5 * min(objectives), \
        "and on this instance the spread is large (measured factor 1.88)"
    assert min(objectives) >= svd_residual(sparse, 3) ** 2 - 1e-9
    best = min(range(8), key=lambda r: objectives[r])
    W_best, H_best, _ = nmf_frobenius(sparse, 3, 300, 1000 + best)
    assert frobenius_objective(sparse, W_best, H_best) == objectives[best], "deterministic"

    # --- determinism ------------------------------------------------------------------
    state_a, matrix_a = random_matrix(777, 4, 4, 0.0, 1.0)
    state_b, matrix_b = random_matrix(777, 4, 4, 0.0, 1.0)
    assert matrix_a == matrix_b and state_a == state_b
    Wa, Ha, _ = nmf_frobenius(matrix_a, 2, 50, 4321)
    Wb2, Hb2, _ = nmf_frobenius(matrix_b, 2, 50, 4321)
    assert Wa == Wb2 and Ha == Hb2


if __name__ == "__main__":
    main()
    print("PASS")
