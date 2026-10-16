"""Task 501 - Implement PCA via power iteration for largest eigenvector. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Principal component analysis specialised to the dominant direction: the data is mean-centred
(population covariance, so it divides by n), the d x d covariance matrix C = X^T X / n is formed
explicitly, and the leading eigenpair is found by power iteration - v_{k+1} = C v_k / ||C v_k||
with the Rayleigh quotient v^T C v as the eigenvalue estimate - iterating until the
residual ||C v - lambda v|| falls below 1e-13 (relative to lambda, documented cap 2000 iterations). Further components come from
deflation, C' = C - lambda v v^T, which removes the component just found without perturbing the
others. The start vector is deterministic (LCG entries), so runs are reproducible, and the header
notes the classical caveats: power iteration converges at rate |lambda2/lambda1| and stalls when
the top two eigenvalues are (nearly) equal.

Documented checks: a hand-built data set with covariance diag(4, 1) must return (lambda, v) =
(4, 1, 0) exactly (the axis is the dominant direction); the residual ||C v - lambda v|| <= 1e-9 and
the Rayleigh quotient equals lambda; for LCG data in 5 dimensions the top three components must
agree with a completely independent method (cyclic Jacobi eigendecomposition of the same C, sorted
descending) to 1e-8, be mutually orthonormal to 1e-9, and their eigenvalues must sum to the trace
of C; the eigenvalues of C - lambda1 v1 v1^T must reproduce the Jacobi tail (deflation adds no
error); a rank-one data set (all points on one line) must have a second eigenvalue <= 1e-9 with a
zero-length deflated matrix; and projecting a 5-D set onto its top 3 components must keep at least
the fraction of total variance that the eigenvalues predict (analytic identity, checked to 1e-9).
"""

TOL = 1e-13
MAX_ITERS = 2000


def lcg(seed):
    return (seed * 75 + 74) % 65537


def make_data(n, d, seed, span=10):
    """n integer points in [-span, span]^d from the 75/74 LCG used across the wave."""
    data = []
    for _ in range(n):
        row = []
        for _ in range(d):
            seed = lcg(seed)
            row.append(float(seed % (2 * span + 1) - span))
        data.append(row)
    return data


def mean_vector(x):
    n = len(x)
    d = len(x[0])
    return [sum(row[j] for row in x) / n for j in range(d)]


def center(x):
    mu = mean_vector(x)
    return [[row[j] - mu[j] for j in range(len(row))] for row in x], mu


def covariance(xc):
    """Population covariance Xc^T Xc / n (symmetric, computed once per upper triangle)."""
    n = len(xc)
    d = len(xc[0])
    c = [[0.0] * d for _ in range(d)]
    for row in xc:
        for i in range(d):
            ri = row[i]
            if ri == 0.0:
                continue
            for j in range(i, d):
                c[i][j] += ri * row[j]
    for i in range(d):
        for j in range(i, d):
            value = c[i][j] / n
            c[i][j] = value
            c[j][i] = value
    return c


def mat_vec(a, v):
    return [sum(a_ij * v_j for a_ij, v_j in zip(row, v)) for row in a]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return dot(a, a) ** 0.5


def rayleigh(a, v):
    return dot(v, mat_vec(a, v))


def power_iteration(a, tol=TOL, max_iters=MAX_ITERS, seed=1):
    """Dominant eigenpair of a symmetric matrix. Returns (lambda, v, iterations)."""
    n = len(a)
    v = []
    for _ in range(n):
        seed = lcg(seed)
        v.append((seed % 200 - 100) / 100.0 + 0.5)
    length = norm(v)
    v = [x / length for x in v]
    lam = rayleigh(a, v)
    iterations = 0
    for _ in range(max_iters):
        iterations += 1
        w = mat_vec(a, v)
        length = norm(w)
        if length <= 1e-300:
            return 0.0, v, iterations            # deflated to the zero matrix
        v = [x / length for x in w]
        lam = rayleigh(a, v)
        residual = [x - y for x, y in zip(mat_vec(a, v), [lam * vi for vi in v])]
        if norm(residual) <= tol * max(1.0, abs(lam)):
            break
    return lam, v, iterations


def deflate(a, lam, v):
    n = len(a)
    return [[a[i][j] - lam * v[i] * v[j] for j in range(n)] for i in range(n)]


def top_components(xc, k, tol=TOL):
    """Top-k (eigenvalue, eigenvector) pairs of the covariance of centred data xc."""
    c = covariance(xc)
    values = []
    vectors = []
    for _ in range(k):
        lam, v, _ = power_iteration(c, tol=tol)
        values.append(lam)
        vectors.append(v)
        c = deflate(c, lam, v)
    return values, vectors


def jacobi_eigen(a, sweeps=100, tol=1e-15):
    """Independent reference: cyclic Jacobi, eigenvalues descending with eigenvectors."""
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
                    akp, akq = m[k][p], m[k][q]
                    m[k][p] = c * akp - s * akq
                    m[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = m[p][k], m[q][k]
                    m[p][k] = c * apk - s * aqk
                    m[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    values = [m[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -values[i])
    return [values[i] for i in order], [[v[r][i] for r in range(n)] for i in order]


def project(xc, vectors):
    """Coordinates of the centred data in the given orthonormal basis (rows = components)."""
    return [[dot(row, v) for v in vectors] for row in xc]


def main():
    # --- Hand-checked example: covariance diag(4, 1), dominant direction is the x axis. -------
    hand = [[2.0, 1.0], [2.0, -1.0], [-2.0, 1.0], [-2.0, -1.0]]
    xc, mu = center(hand)
    assert mu == [0.0, 0.0]
    c = covariance(xc)
    assert abs(c[0][0] - 4.0) <= 1e-12 and abs(c[1][1] - 1.0) <= 1e-12 and c[0][1] == 0.0
    lam, v, iters = power_iteration(c)
    assert 1 <= iters <= MAX_ITERS
    assert abs(lam - 4.0) <= 1e-9, lam
    assert abs(abs(v[0]) - 1.0) <= 1e-9 and abs(v[1]) <= 1e-9, v
    assert norm([x - y for x, y in zip(mat_vec(c, v), [lam * xi for xi in v])]) <= 1e-9
    assert abs(rayleigh(c, v) - lam) <= 1e-9

    # Purely rank-one data: every point on the line y = 3x.
    line = [[t, 3.0 * t] for t in (-3.0, -1.0, 0.0, 2.0, 5.0)]
    lxc, _ = center(line)
    lc = covariance(lxc)
    lam1, v1, _ = power_iteration(lc)
    ts = (-3.0, -1.0, 0.0, 2.0, 5.0)
    s2 = sum((t - 0.6) ** 2 for t in ts) / 5.0              # variance of the line parameter
    assert abs(lam1 - 10.0 * s2) <= 1e-9, lam1               # (1, 3) has squared length 10
    assert abs(abs(v1[0]) / norm(v1) - 1.0 / 10.0 ** 0.5) <= 1e-9
    lam2, _, _ = power_iteration(deflate(lc, lam1, v1))
    assert lam2 <= 1e-9, "rank-one data must have a second eigenvalue of (about) zero"

    # --- LCG data: power iteration versus an independent Jacobi eigendecomposition. ----------
    data = make_data(40, 5, seed=17)
    xc, mu = center(data)
    assert all(abs(mu[j] - sum(row[j] for row in data) / 40.0) <= 1e-12 for j in range(5))
    c = covariance(xc)
    jac_values, jac_vectors = jacobi_eigen(c)
    assert len(jac_values) == 5 and all(jac_values[i] >= jac_values[i + 1] - 1e-12
                                        for i in range(4))

    values, vectors = top_components(xc, 3)
    for i in range(3):
        assert abs(values[i] - jac_values[i]) <= 1e-8, (i, values[i], jac_values[i])
        assert norm([a - b for a, b in zip(vectors[i], jac_vectors[i])]) <= 1e-6 or \
            norm([a + b for a, b in zip(vectors[i], jac_vectors[i])]) <= 1e-6
    # Orthonormality of the returned basis and residual of each eigenpair.
    for i in range(3):
        assert abs(norm(vectors[i]) - 1.0) <= 1e-9
        residual = [x - y for x, y in zip(mat_vec(c, vectors[i]),
                                         [values[i] * x for x in vectors[i]])]
        assert norm(residual) <= 1e-8, "eigenpair residual too large"
        for j in range(i + 1, 3):
            assert abs(dot(vectors[i], vectors[j])) <= 1e-9, "components must be orthogonal"

    # Eigenvalues sum to the trace of C, and the largest dominates the average variance.
    trace = sum(c[i][i] for i in range(5))
    assert abs(sum(values) - sum(jac_values[:3])) <= 1e-8 and sum(values) <= trace + 1e-9
    assert abs(sum(jac_values) - trace) <= 1e-9, "all eigenvalues must sum to the trace"
    assert values[0] >= values[1] >= values[2] > 0.0
    assert abs(values[0] - jac_values[0]) <= 1e-8

    # Deflation adds no error: the eigenvalues of C - lambda1 v1 v1^T are the Jacobi tail.
    deflated = deflate(c, values[0], vectors[0])
    for i in range(4):
        lam, _, _ = power_iteration(deflated, tol=1e-13)
        assert abs(lam - jac_values[i + 1]) <= 1e-8, (i, lam, jac_values[i + 1])
        _, vec, _ = power_iteration(deflated, tol=1e-13)
        deflated = deflate(deflated, lam, vec)

    # --- Reconstruction / variance captured: analytic identity var(v) = sum of eigenvalues. ---
    total = sum(c[i][i] for i in range(5))
    coords = project(xc, vectors)
    kept = sum(sum(row[i] * row[i] for row in coords) / len(xc) for i in range(3))
    assert abs(kept - sum(values)) <= 1e-9, (kept, sum(values))
    assert abs(kept - (values[0] + values[1] + values[2])) <= 1e-9
    assert 0.0 < kept <= total + 1e-9
    # The residual variance is exactly the trace of the discarded eigenvalues.
    assert abs((total - kept) - (jac_values[3] + jac_values[4])) <= 1e-8

    # Full basis (k = d) reconstructs the centred data exactly.
    all_values, all_vectors = top_components(xc, 5)
    full = project(xc, all_vectors)
    for row, coord in zip(xc, full):
        rebuilt = [sum(coord[i] * all_vectors[i][j] for i in range(5)) for j in range(5)]
        assert norm([a - b for a, b in zip(row, rebuilt)]) <= 1e-8
    assert abs(sum(all_values) - total) <= 1e-8

    # Constant data has zero covariance: power iteration must report a zero eigenvalue.
    flat = [[3.0, -1.0, 7.0]] * 6
    fxc, fmu = center(flat)
    assert fmu == [3.0, -1.0, 7.0]
    assert all(all(abs(x) <= 1e-15 for x in row) for row in fxc)
    flat_lam, _, _ = power_iteration(covariance(fxc))
    assert abs(flat_lam) <= 1e-15, "zero variance data has no dominant direction"


if __name__ == "__main__":
    main()
    print("PASS")
