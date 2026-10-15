"""Task 500 - Implement spectral clustering small example (eigen decomposition). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Spectral 2-clustering of a handful of points: build the Gaussian similarity matrix
W_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)), form the combinatorial Laplacian L = D - W (D the
row sums), and take the Fiedler vector - the eigenvector of the second-smallest eigenvalue -
via cyclic Jacobi rotations (sweeps until the off-diagonal Frobenius norm drops below 1e-12,
documented cap 100 sweeps). The split threshold is the mean of the Fiedler vector, with the
sign fixed by making point 0 lie at or above the mean. Documented checks: the zero eigenvalue
and the residual ||L v - lambda2 v|| (<= 1e-9), lambda2 < lambda3 with the vector orthogonal
to the constant, and the resulting cut equals the minimum over all balanced 6-6 cuts of the
canonical two-blob example (the planted split). The canonical example is two six-point
clusters around (0,0) and (10,10) with sigma = 3.0 (cross-cluster similarities ~1e-4, so
lambda2 ~ 4.85e-4 - comfortably above the Jacobi tolerance); the tiny second example is two
LCG-jittered
4-point blobs around (0,0) and (3,0) with sigma = 0.5, where the Fiedler sign pattern must
again reproduce the planted partition.
"""
import math

CANONICAL = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 1.5), (1.5, 0.5),
             (10.0, 10.0), (11.0, 10.0), (10.0, 11.0), (11.0, 11.0), (10.5, 10.5), (9.5, 10.5)]
CANONICAL_TRUTH = [0] * 6 + [1] * 6
SIGMA_CANONICAL = 3.0
SIGMA_TINY = 0.5


def sqdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def tiny_blobs(seed):
    """Two 4-point blobs around (0,0) and (3,0), jitter +/-0.4 in 0.01 steps via the LCG."""
    points = []
    for cx in (0.0, 3.0):
        for _ in range(4):
            seed = (seed * 75 + 74) % 65537
            x = cx + (seed % 81 - 40) / 100.0
            seed = (seed * 75 + 74) % 65537
            y = (seed % 81 - 40) / 100.0
            points.append((x, y))
    return points, [0] * 4 + [1] * 4


def similarity(points, sigma):
    n = len(points)
    w = [[0.0] * n for _ in range(n)]
    denom = 2.0 * sigma * sigma
    for i in range(n):
        for j in range(i + 1, n):
            value = math.exp(-sqdist(points[i], points[j]) / denom)
            w[i][j] = value
            w[j][i] = value
    return w


def laplacian(w):
    n = len(w)
    return [[(sum(w[i]) if i == j else 0.0) - w[i][j] for j in range(n)] for i in range(n)]


def jacobi_eigh(matrix, max_sweeps=100, tol=1e-12):
    """Cyclic Jacobi eigen-decomposition of a symmetric matrix; returns (values, vectors)
    with eigenvectors as columns. Returns the number of sweeps performed as well."""
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    sweeps = 0
    for _ in range(max_sweeps):
        off = math.sqrt(sum(a[i][j] * a[i][j] for i in range(n) for j in range(i + 1, n)))
        if off <= tol:
            break
        sweeps += 1
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = 1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0))
                if theta < 0.0:
                    t = -t
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    if k == p or k == q:
                        continue
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                    a[p][k] = a[k][p]
                    a[q][k] = a[k][q]
                app, aqq, apq = a[p][p], a[q][q], a[p][q]
                a[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
                a[q][q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
                a[p][q] = 0.0
                a[q][p] = 0.0
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v, sweeps


def spectral_split(points, sigma):
    """Returns (labels, lambda2, residual, orthogonality error, sweeps)."""
    n = len(points)
    w = similarity(points, sigma)
    lmat = laplacian(w)
    values, vectors, sweeps = jacobi_eigh(lmat)
    order = sorted(range(n), key=lambda i: (values[i], i))
    fiedler = order[1]
    lam2 = values[fiedler]
    vec = [vectors[i][fiedler] for i in range(n)]
    mean = sum(vec) / n
    if vec[0] < mean:
        vec = [-x for x in vec]
    labels = [0 if vec[i] >= mean else 1 for i in range(n)]
    residual = math.sqrt(sum((sum(lmat[i][j] * vec[j] for j in range(n))
                              - lam2 * vec[i]) ** 2 for i in range(n)))
    orthogonality = abs(sum(vec))
    return labels, lam2, residual, orthogonality, sweeps


def cut_weight(w, side):
    n = len(w)
    total = 0.0
    for i in range(n):
        for j in range(n):
            if side[i] == 0 and side[j] == 1:
                total += w[i][j]
    return total


def min_balanced_cut(w, half):
    """Brute force over all balanced half-half splits (mirror duplicates cheap to skip)."""
    from itertools import combinations
    n = len(w)
    best = None
    rest = list(range(n))
    for combo in combinations(range(1, n), half - 1):
        side = [1] * n
        side[0] = 0
        for i in combo:
            side[i] = 0
        if side[0] == side[n - 1]:
            continue
        weight = cut_weight(w, side)
        if best is None or weight < best:
            best = weight
    return best


def main():
    # Jacobi sanity on a 2x2 textbook matrix: eigenvalues {1, 3}, residual ~ 0.
    values, vectors, _ = jacobi_eigh([[2.0, 1.0], [1.0, 2.0]])
    order = sorted(range(2), key=lambda i: values[i])
    assert abs(values[order[0]] - 1.0) <= 1e-12 and abs(values[order[1]] - 3.0) <= 1e-12

    # Canonical two-blob example.
    labels, lam2, residual, orthogonality, sweeps = spectral_split(CANONICAL, SIGMA_CANONICAL)
    assert residual <= 1e-9, "eigen-pair residual"
    assert orthogonality <= 1e-8, "Fiedler vector is orthogonal to the constant"
    assert 1e-6 <= lam2 <= 1e-2, "two separated blobs: tiny positive Fiedler value"
    assert sweeps <= 100

    # lambda2 < lambda3 for the canonical example (requires the full spectrum).
    w = similarity(CANONICAL, SIGMA_CANONICAL)
    values, _, _ = jacobi_eigh(laplacian(w))
    asc = sorted(values)
    assert abs(asc[0]) <= 1e-9, "connected graph: one zero eigenvalue"
    assert asc[1] < asc[2], "Fiedler eigenvalue is strictly second smallest"
    assert asc[1] <= 1e-2 * asc[2], "documented: the Fiedler value separates the blobs"

    # The split reproduces the planted partition and achieves the minimum balanced cut.
    assert labels == CANONICAL_TRUTH, "canonical split must match the planted structure"
    spectral_cut = cut_weight(w, labels)
    best_balanced = min_balanced_cut(w, 6)
    assert abs(spectral_cut - best_balanced) <= 1e-9, "spectral cut equals the min 6-6 cut"
    # Cross-blob similarity is astronomically small: the cut is far below the within mass.
    assert spectral_cut <= 1e-3 * sum(sum(row) for row in w)

    # Tiny LCG blobs: same determinism, planted structure recovered at sigma = 0.5.
    tiny, truth = tiny_blobs(2024)
    tiny_labels, tiny_lam2, tiny_residual, tiny_orth, _ = spectral_split(tiny, SIGMA_TINY)
    assert tiny_labels == truth, "tiny LCG blobs split correctly"
    assert tiny_residual <= 1e-9 and tiny_orth <= 1e-8
    assert 1e-8 <= tiny_lam2 <= 1e-2, "tiny positive Fiedler value"
    repeat_labels, _, _, _, _ = spectral_split(tiny, SIGMA_TINY)
    assert repeat_labels == tiny_labels, "deterministic re-run"

    # Same tiny blobs spaced far apart must not be split at all (heavier sigma): the split is
    # still the planted one because the within-blob mass dominates either way.
    spaced = [(x * 10.0, y * 10.0) for x, y in tiny]
    spaced_labels, _, _, _, _ = spectral_split(spaced, SIGMA_TINY * 10.0)
    assert spaced_labels == truth


if __name__ == "__main__":
    main()
    print("PASS")
