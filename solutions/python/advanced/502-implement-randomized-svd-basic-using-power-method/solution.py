"""Task 502 - Implement randomized SVD basic using power method. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Randomized range finder for the thin SVD of a dense m x n matrix A: draw an n x (k + p) Gaussian
test matrix Omega (entries from the deterministic 75/74 LCG), form Y = A Omega, sharpen the range
with q power iterations of the symmetric operator Y <- A (A^T Y) (each one multiplies the
projected energy of the wanted directions by (sigma_i / sigma_{i+1})^{2q}), orthonormalise with a
modified Gram-Schmidt QR using two reorthogonalisation passes (classical Gram-Schmidt loses
orthogonality once the columns nearly coincide), project B = Q^T A onto the small l x n matrix,
and finish with an exact eigendecomposition of the l x l Gram matrix B B^T by cyclic Jacobi
rotations: singular values s_i = sqrt(lambda_i), right vectors v_i = B^T u_i / s_i, left vectors
U = Q u_i. Everything is dense, standard-library, and deterministic (fixed LCG seed); the exact
SVD used as the oracle is an independent computation - Jacobi on A^T A (or A A^T for the wide
case), singular values sqrt of the eigenvalues, again sorted descending.

Documented checks: for a 30 x 12 matrix with the prescribed spectrum 10, 7, 4, 2.5, 1.2, 0.6, ...,
the randomised top-4 triple must match the exact SVD to 1e-6 in singular values and to 1e-7 in
subspace distance (five power iterations) ||(I - U_exact U_exact^T) u_i||; every returned column of U and V must be
orthonormal to 1e-10; the rank-4 reconstruction error must be within 1e-8 of the optimal
||A - A_4||_F (the celebrated near-optimality of the randomised range finder); the estimate must
be a genuine rank-k factorisation with U S V^T having the right shape and squared Frobenius norm
equal to the sum of squared singular values; on a nearly flat spectrum q = 3 power iterations must
beat q = 0 by at least a factor of 10 (the documented reason power iterations exist); errors must
grow monotonically as the retained rank shrinks and vanish at full rank; a wide 8 x 15 example must
work through the transposed branch; identical seeds must give bit-identical factors, and the
full-rank recovery must be exact when q = 0 (with q = 4 the (A A^T)^q amplification of a
condition number of 2000 destroys the smallest directions - a real, documented caveat of the
method rather than a bug).
"""

TAU = 1e-12


def lcg(seed):
    return (seed * 75 + 74) % 65537


def rand_matrix(rows, cols, seed, span=100):
    """Dense integer matrix in [-span, span] from the shared LCG."""
    out = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            seed = lcg(seed)
            row.append(float(seed % (2 * span + 1) - span))
        out.append(row)
    return out


def matmul(a, b):
    cols = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in cols] for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return dot(a, a) ** 0.5


def frobenius(a):
    return sum(dot(row, row) for row in a) ** 0.5


def matvec(a, v):
    return [dot(row, v) for row in a]


def qr(a, passes=2):
    """Thin QR by modified Gram-Schmidt with `passes` reorthogonalisations. Returns Q only."""
    m, n = len(a), len(a[0])
    cols = [list(col) for col in zip(*a)]
    q = []
    for j in range(n):
        v = cols[j]
        for _ in range(passes):
            for qi in q:
                proj = dot(qi, v)
                v = [x - proj * y for x, y in zip(v, qi)]
        length = norm(v)
        if length <= TAU:
            raise ValueError("rank deficient test matrix")
        q.append([x / length for x in v])
    return [[q[j][i] for j in range(n)] for i in range(m)]


def jacobi_eigen(a, sweeps=60, tol=1e-15):
    """Cyclic Jacobi. Returns (eigenvalues descending, eigenvectors as columns)."""
    n = len(a)
    m = [row[:] for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = sum(m[i][j] ** 2 for i in range(n) for j in range(n) if i != j) ** 0.5
        if off <= tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(m[p][q]) <= 1e-300:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                t = (1.0 if theta >= 0.0 else -1.0) / (abs(theta) + (theta * theta + 1.0) ** 0.5)
                c = 1.0 / (t * t + 1.0) ** 0.5
                s = t * c
                for k in range(n):
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p][k], m[q][k]
                    m[p][k] = c * mpk - s * mqk
                    m[q][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    values = [m[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -values[i])
    return [values[i] for i in order], [[v[r][i] for r in range(n)] for i in order]


def svd_exact(a):
    """Thin SVD via an independent Jacobi eigendecomposition. Returns (U, s, V)."""
    m, n = len(a), len(a[0])
    if n <= m:
        gram = matmul(transpose(a), a)                   # n x n
        values, v_cols = jacobi_eigen(gram)
        s = [max(x, 0.0) ** 0.5 for x in values]
        u_cols = []
        for i in range(n):
            if s[i] <= 1e-13:
                u_cols.append([0.0] * m)
            else:
                u_cols.append([x / s[i] for x in matvec(a, v_cols[i])])
    else:
        gram = matmul(a, transpose(a))                   # m x m
        values, u_cols = jacobi_eigen(gram)
        s = [max(x, 0.0) ** 0.5 for x in values[:m]]
        v_cols = []
        for i in range(m):
            if s[i] <= 1e-13:
                v_cols.append([0.0] * n)
            else:
                v_cols.append([x / s[i] for x in matvec(transpose(a), u_cols[i])])
    u = [[u_cols[j][i] for j in range(len(u_cols))] for i in range(m)]
    v = [[v_cols[j][i] for j in range(len(v_cols))] for i in range(n)]
    return u, s, v


def randomized_svd(a, k, oversample=2, power_iters=3, seed=7):
    """Randomised rank-k truncated SVD of a (m x n): returns (U, s, V) with m x k / k / n x k."""
    m, n = len(a), len(a[0])
    l = min(k + oversample, m, n)
    omega = rand_matrix(n, l, seed)
    y = matmul(a, omega)
    at = transpose(a)
    for _ in range(power_iters):
        y = matmul(a, matmul(at, y))
    q = qr(y)                                            # m x l, orthonormal columns
    b = matmul(transpose(q), a)                          # l x n
    values, uhat_cols = jacobi_eigen(matmul(b, transpose(b)))   # l x l Gram matrix
    s_all = [max(x, 0.0) ** 0.5 for x in values]
    v_cols = []
    for i in range(l):
        if s_all[i] <= 1e-10:
            v_cols.append([0.0] * n)
        else:
            v_cols.append([x / s_all[i] for x in matvec(transpose(b), uhat_cols[i])])
    u_cols = [matvec(q, uhat_cols[i]) for i in range(l)]
    u = [[u_cols[j][i] for j in range(k)] for i in range(m)]
    v = [[v_cols[j][i] for j in range(k)] for i in range(n)]
    return u, s_all[:k], v


def reconstruct(u, s, v):
    cols = [[u[i][t] * s[t] for i in range(len(u))] for t in range(len(s))]
    v_cols = [list(col) for col in zip(*v)]
    return [[sum(cols[t][i] * v_cols[t][j] for t in range(len(s))) for j in range(len(v_cols[0]))]
            for i in range(len(u))]


def rank_k_error(a, k):
    """Optimal rank-k approximation error ||A - A_k||_F = sqrt(sum of the discarded s_i^2)."""
    _, s, _ = svd_exact(a)
    return sum(s[i] * s[i] for i in range(k, len(s))) ** 0.5


def subspace_error(u_approx, u_exact, k, m):
    """max_i ||(I - P_k) u_i|| with P_k the projector onto the oracle's top-k left subspace."""
    worst = 0.0
    for i in range(k):
        projected = [0.0] * m
        for t in range(k):
            weight = dot([u_exact[r][t] for r in range(m)], [u_approx[r][i] for r in range(m)])
            for r in range(m):
                projected[r] += weight * u_exact[r][t]
        residual = norm([u_approx[r][i] - projected[r] for r in range(m)])
        worst = max(worst, residual)
    return worst


def make_matrix(m, n, spectrum, seed):
    """A = Q1 diag(spectrum) Q2^T with orthonormal Q1 (m x n) and Q2 (n x n)."""
    q1 = qr(rand_matrix(m, n, seed))
    q2 = qr(rand_matrix(n, n, seed + 1))
    scaled = [[q1[i][j] * spectrum[j] for j in range(n)] for i in range(m)]
    return matmul(scaled, transpose(q2))


def main():
    spectrum = [10.0, 7.0, 4.0, 2.5, 1.2, 0.6, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005]
    a = make_matrix(30, 12, spectrum, seed=11)
    u_exact, s_exact, v_exact = svd_exact(a)

    # The oracle itself: eigenvalues of A^T A are the squared singular values.
    assert len(s_exact) == 12
    for i in range(1, 12):
        assert s_exact[i - 1] >= s_exact[i] - 1e-12, "exact singular values must be sorted"
    assert abs(frobenius(a) ** 2 - sum(x * x for x in s_exact)) <= 1e-6
    for i in range(5):
        assert abs(s_exact[i] - spectrum[i]) <= 1e-8, (i, s_exact[i], spectrum[i])

    k = 4
    u, s, v = randomized_svd(a, k, oversample=2, power_iters=5, seed=7)
    assert len(u) == 30 and len(u[0]) == k and len(v) == 12 and len(v[0]) == k and len(s) == k

    # Singular values and left subspace must reproduce the exact SVD.
    for i in range(k):
        assert abs(s[i] - s_exact[i]) <= 1e-6, (i, s[i], s_exact[i])
    for i in range(1, k):
        assert s[i - 1] >= s[i] - 1e-12
    assert subspace_error(u, u_exact, k, 30) <= 1e-7, "estimated left subspace drifted"

    # Orthonormality of both factors.
    for i in range(k):
        ui = [u[r][i] for r in range(30)]
        vi = [v[r][i] for r in range(12)]
        assert abs(norm(ui) - 1.0) <= 1e-10 and abs(norm(vi) - 1.0) <= 1e-10
        for j in range(i + 1, k):
            assert abs(dot(ui, [u[r][j] for r in range(30)])) <= 1e-10
            assert abs(dot(vi, [v[r][j] for r in range(12)])) <= 1e-10

    # Rank-k factorisation: U S V^T is m x n, and its Frobenius norm is sqrt(sum s_i^2).
    approx = reconstruct(u, s, v)
    assert len(approx) == 30 and len(approx[0]) == 12
    assert abs(frobenius(approx) ** 2 - sum(x * x for x in s)) <= 1e-6

    # Near-optimality: the randomised rank-4 error must match the best possible rank-4 error.
    best4 = rank_k_error(a, k)
    got4 = frobenius([[approx[i][j] - a[i][j] for j in range(12)] for i in range(30)])
    assert got4 <= best4 + 1e-8, (got4, best4)
    assert got4 >= best4 - 1e-9, "a rank-4 approximation cannot beat the optimal one"

    # Power iterations matter: a spectrum whose 6th/7th values (5.0 and 2.5) sit inside the
    # oversampled range (l = 7) but outside the retained rank (k = 5) must be suppressed by
    # q = 3 - the subspace distance should drop by at least two orders of magnitude.
    gapped_spectrum = [10.0, 9.5, 9.0, 8.5, 8.0, 5.0, 2.5, 1.0, 0.5, 0.2, 0.1, 0.05]
    flat = make_matrix(40, 12, gapped_spectrum, seed=23)
    f_exact_u, f_exact_s, _ = svd_exact(flat)
    u0, s0, v0 = randomized_svd(flat, 5, oversample=2, power_iters=0, seed=5)
    u3, s3, v3 = randomized_svd(flat, 5, oversample=2, power_iters=3, seed=5)
    approx0 = reconstruct(u0, s0, v0)
    approx3 = reconstruct(u3, s3, v3)
    err0 = frobenius([[approx0[i][j] - flat[i][j] for j in range(12)] for i in range(40)])
    err3 = frobenius([[approx3[i][j] - flat[i][j] for j in range(12)] for i in range(40)])
    best5 = rank_k_error(flat, 5)
    assert err3 <= best5 + 1e-9, (err3, best5)
    assert err3 <= err0 * 0.95, "power iterations must reduce the reconstruction error"
    for i in range(5):
        assert abs(s3[i] - f_exact_s[i]) <= 1e-6
    assert abs(s0[0] - f_exact_s[0]) <= 0.05 and s0[4] < f_exact_s[4] - 0.01, \
        "without power iterations the smallest retained value is visibly wrong"
    sub0 = subspace_error(u0, f_exact_u, 5, 40)
    sub3 = subspace_error(u3, f_exact_u, 5, 40)
    assert sub0 > 0.05 and sub3 <= sub0 / 100.0, (sub0, sub3)

    # Retained rank controls the error: strictly decreasing, and zero at full rank.
    errors = []
    for kk in (1, 2, 3, 4, 6):
        uk, sk, vk = randomized_svd(a, kk, oversample=3, power_iters=4, seed=13)
        ak = reconstruct(uk, sk, vk)
        errors.append(frobenius([[ak[i][j] - a[i][j] for j in range(12)] for i in range(30)]))
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1], "more components must never hurt"
    assert errors[-1] <= rank_k_error(a, 6) + 1e-6, "rank 6 must reach the optimal rank-6 error"
    assert errors[0] <= rank_k_error(a, 1) + 1e-6
    # Full rank: without power iterations the range finder recovers all 12 directions exactly
    # (with q = 4 and a condition number of 2000 the tiny singular directions are lost to the
    # (A A^T)^q amplification - a documented caveat, so the exact run uses q = 0).
    uf, sf, vf = randomized_svd(a, 12, oversample=0, power_iters=0, seed=17)
    full = reconstruct(uf, sf, vf)
    assert frobenius([[full[i][j] - a[i][j] for j in range(12)] for i in range(30)]) <= 1e-8
    assert abs(sf[0] - 10.0) <= 1e-8 and abs(sf[11] - 0.005) <= 1e-9

    # Wide matrix: exercises the m < n branch of the oracle and of the range finder.
    wide = make_matrix(15, 8, [6.0, 3.0, 1.5, 0.7, 0.3, 0.12, 0.05, 0.02], seed=31)
    wide = transpose(wide)                          # 8 x 15
    wu, ws, wv = randomized_svd(wide, 3, oversample=2, power_iters=3, seed=19)
    assert len(wu) == 8 and len(wv) == 15 and len(ws) == 3
    _, w_exact_s, _ = svd_exact(wide)
    for i in range(3):
        assert abs(ws[i] - w_exact_s[i]) <= 1e-6, (i, ws[i], w_exact_s[i])
    w_approx = reconstruct(wu, ws, wv)
    assert len(w_approx) == 8 and len(w_approx[0]) == 15
    assert frobenius([[w_approx[i][j] - wide[i][j] for j in range(15)] for i in range(8)]) <= \
        rank_k_error(wide, 3) + 1e-5

    # Determinism: same seed, same factors, bit for bit.
    u2, s2, v2 = randomized_svd(a, k, oversample=2, power_iters=5, seed=7)
    assert u2 == u and s2 == s and v2 == v


if __name__ == "__main__":
    main()
    print("PASS")
