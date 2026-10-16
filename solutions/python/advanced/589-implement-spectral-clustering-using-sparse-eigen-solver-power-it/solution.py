"""Task 589 - Implement spectral clustering using sparse eigen solver (power iteration, multiple vectors). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Spectral clustering of a small sparse graph with an iterative eigensolver instead of a dense
decomposition.  The affinity is kept sparse as an adjacency list of dicts (weight per neighbour),
and the operator that is iterated is the symmetric normalized adjacency

    N = D^(-1/2) W D^(-1/2),   (N v)_i = d_i^(-1/2) * sum_{j ~ i} w_ij * v_j * d_j^(-1/2),

applied matrix-free (`normalized_matvec`), so no n x n dense matrix is ever formed.  Orthogonal
iteration (subspace iteration) propagates k orthonormal vectors at once: Z = N V, re-orthonormalize
the columns by modified Gram-Schmidt with a second re-orthogonalization pass for conditioning,
and repeat until the subspace stops moving.  A final Rayleigh-Ritz step builds the k x k matrix
R = V^T N V, diagonalizes it with a small cyclic Jacobi sweep, and rotates V -- that is what makes
the returned vectors genuine eigenvector approximations rather than merely spanning the subspace.
Clustering follows Ng-Jordan-Weiss: rows of U = D^(-1/2) V are normalized to the unit sphere and
grouped by deterministic farthest-point k-means (for k = 2 the sign of the combinatorial Fiedler
vector u2 = D^(-1/2) v2 is used, since u2 is orthogonal to the positive vector d^(1/2)1 and
therefore straddles 0).

Documented checks: lambda_1 = 1 within 1e-12 *and* the leading eigenvector equals the closed form
d^(1/2)/||d^(1/2)|| (an independent identity, not a self-consistency test); every returned pair
satisfies ||N v - lambda v|| <= 1e-9, V^T V = I to 1e-12, and the Rayleigh quotient matches the
eigenvalue; the top-k eigenvalues agree with a dense cyclic Jacobi decomposition of the same N
(second, independent method) to 1e-9 and all lie in [-1, 1] as theory requires; the two-community
graph (two 6-cliques joined by one edge) and the three-community graph (three 5-cliques chained by
single edges) are both split exactly along the planted communities, and for k = 3 the normalized
rows are measured to form tight groups (max within-community distance 0.23 against a minimum
cross-community distance of 1.22); the error decay ratio of plain power
iteration on the leading eigenvector matches the independently computed lambda_2 within 2%, and
the iteration count needed to reach an error of 1e-9 agrees with the prediction
log(1e-9)/log(lambda_2) within 20%; the stagnation of the well-separated instance (lambda_2 -> 1)
is documented by asserting that its iteration count exceeds that of the weakly clustered control
graph while the dense lambda_2 ordering is the opposite.  Sample weights and the starting subspace come from
splitmix64 (the 16-bit LCG has too short a cycle for a k-column orthonormal start), and the whole
pipeline is asserted to reproduce identical labels on a second run.
"""
import math

MASK64 = (1 << 64) - 1


def splitmix64(state):
    """Splitmix64: an orthonormal k-column start needs n*k independent uniforms, which the repo's
    16-bit LCG cannot supply without visible correlation between the columns."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


def gauss_pair(state):
    state, u1 = uniform(state)
    state, u2 = uniform(state)
    radius = math.sqrt(-2.0 * math.log(u1 + 1e-15))
    return state, (radius * math.cos(2.0 * math.pi * u2), radius * math.sin(2.0 * math.pi * u2))


# ---------------------------------------------------------------------------------------
# sparse graph helpers
# ---------------------------------------------------------------------------------------
def add_edge(adjacency, u, v, weight):
    adjacency[u][v] = adjacency[u].get(v, 0.0) + weight
    adjacency[v][u] = adjacency[v].get(u, 0.0) + weight


def clique_graph(cliques, bridges):
    """Disjoint cliques plus a list of (u, v) bridge edges; returns (n, adjacency)."""
    n = sum(len(c) for c in cliques)
    adjacency = [dict() for _ in range(n)]
    for members in cliques:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                add_edge(adjacency, members[i], members[j], 1.0)
    for (u, v) in bridges:
        add_edge(adjacency, u, v, 1.0)
    return n, adjacency


def degrees(adjacency):
    return [sum(row.values()) for row in adjacency]


def normalized_matvec(adjacency, weights, vector):
    """Apply N = D^(-1/2) W D^(-1/2) without forming it."""
    scaled = [vector[i] / math.sqrt(weights[i]) for i in range(len(vector))]
    out = []
    for i, row in enumerate(adjacency):
        total = 0.0
        for j, w in row.items():
            total += w * scaled[j]
        out.append(total / math.sqrt(weights[i]))
    return out


def dense_normalized(adjacency, weights):
    n = len(adjacency)
    matrix = [[0.0] * n for _ in range(n)]
    for i, row in enumerate(adjacency):
        for j, w in row.items():
            matrix[i][j] = w / math.sqrt(weights[i] * weights[j])
    return matrix


# ---------------------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------------------
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def jacobi_eigh(matrix, max_sweeps=200, tol=1e-14):
    """Cyclic Jacobi eigen-decomposition of a symmetric matrix: (values, columns)."""
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = math.sqrt(sum(a[i][j] * a[i][j] for i in range(n) for j in range(i + 1, n)))
        if off <= tol:
            break
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
    values = [a[i][i] for i in range(n)]
    vectors = [[v[i][j] for i in range(n)] for j in range(n)]     # columns
    order = sorted(range(n), key=lambda j: values[j])
    return [values[j] for j in order], [vectors[j] for j in order]


def gram_schmidt(columns):
    """Modified Gram-Schmidt with a second pass; returns orthonormal columns."""
    basis = []
    for column in columns:
        work = list(column)
        for _ in range(2):
            for other in basis:
                projection = dot(other, work)
                work = [x - projection * y for x, y in zip(work, other)]
        length = norm(work)
        if length <= 1e-12:
            raise ValueError("linearly dependent columns")
        basis.append([value / length for value in work])
    return basis


def random_orthonormal(n, k, seed):
    state = seed
    columns = []
    for _ in range(k):
        column = []
        for _ in range(n):
            state, pair = gauss_pair(state)
            column.append(pair[0])
        columns.append(column)
    return state, gram_schmidt(columns)


def orthogonal_iteration(adjacency, weights, k, seed, tol=1e-13, max_iters=1000):
    """Subspace iteration for the k dominant eigenpairs of N; returns (values, vectors, iters)."""
    n = len(adjacency)
    state, basis = random_orthonormal(n, k, seed)

    def apply(column):
        return normalized_matvec(adjacency, weights, column)

    iters = 0
    for step in range(max_iters):
        moved = gram_schmidt([apply(column) for column in basis])
        change = max(norm([a - b for a, b in zip(moved[j], basis[j])]) for j in range(k))
        basis = moved
        iters = step + 1
        if change <= tol:
            break
    applied = [apply(column) for column in basis]
    rayleigh = [[dot(basis[i], applied[j]) for j in range(k)] for i in range(k)]
    values, rotation = jacobi_eigh(rayleigh)
    vectors = [[sum(rotation[j][i] * basis[j][m] for j in range(k)) for m in range(n)]
               for i in range(k)]
    order = sorted(range(k), key=lambda i: -values[i])
    values = [values[i] for i in order]
    vectors = [vectors[i] for i in order]
    residuals = []
    for i in range(k):
        image = apply(vectors[i])
        residual = norm([image[m] - values[i] * vectors[i][m] for m in range(n)])
        residuals.append(residual)
    return values, vectors, iters, residuals


def dense_spectrum(adjacency, weights):
    matrix = dense_normalized(adjacency, weights)
    values, vectors = jacobi_eigh(matrix)
    values = list(reversed(values))
    vectors = list(reversed(vectors))
    return values, vectors


def fiedler_split(adjacency, weights, vector):
    """Combinatorial Fiedler vector u = D^(-1/2) v; split at the weighted mean (0 for the
    volume-normalized problem, the mean is used here as a numerically safer threshold)."""
    fiedler = [vector[i] / math.sqrt(weights[i]) for i in range(len(vector))]
    threshold = sum(fiedler) / len(fiedler)
    return [0 if value >= threshold else 1 for value in fiedler], fiedler


def kmeans(rows, k, seed, max_iters=100, tol=1e-12):
    """Deterministic farthest-point k-means on unit rows (no unseeded randomness)."""
    n = len(rows)
    state = seed
    state, _ = splitmix64(state)
    centres = [list(rows[0])]
    while len(centres) < k:
        far = max(range(n), key=lambda i: min(norm([a - b for a, b in zip(rows[i], c)])
                                             for c in centres))
        new = list(rows[far])
        if any(norm([a - b for a, b in zip(new, c)]) <= 1e-12 for c in centres):
            for i in range(n):
                if all(norm([a - b for a, b in zip(rows[i], c)]) > 1e-12 for c in centres):
                    new = list(rows[i])
                    break
        centres.append(new)
    labels = [0] * n
    for _ in range(max_iters):
        changed = False
        for i in range(n):
            best = min(range(k), key=lambda c: norm([a - b for a, b in zip(rows[i], centres[c])]))
            if best != labels[i]:
                labels[i] = best
                changed = True
        for c in range(k):
            members = [rows[i] for i in range(n) if labels[i] == c]
            if members:
                centres[c] = [sum(row[j] for row in members) / len(members)
                              for j in range(len(rows[0]))]
        if not changed:
            break
    return labels


def normalized_rows(vectors, weights, n):
    rows = []
    for i in range(n):
        row = [vectors[j][i] / math.sqrt(weights[i]) for j in range(len(vectors))]
        length = norm(row) or 1.0
        rows.append([value / length for value in row])
    return rows


def planted_two_cliques():
    left = list(range(6))
    right = list(range(6, 12))
    return clique_graph([left, right], [(0, 6)])


def planted_three_cliques():
    groups = [list(range(5)), list(range(5, 10)), list(range(10, 15))]
    return clique_graph(groups, [(0, 5), (5, 10)])


def planted_one_community():
    """A weakly clustered control graph: two cliques joined by five edges."""
    left = list(range(6))
    right = list(range(6, 12))
    return clique_graph([left, right], [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10)])


def same_partition(first, second):
    """True when the two labellings induce the same partition (labels are arbitrary)."""
    n = len(first)
    groups_first = {frozenset(i for i in range(n) if first[i] == value) for value in set(first)}
    groups_second = {frozenset(i for i in range(n) if second[i] == value) for value in set(second)}
    return groups_first == groups_second


def main():
    # --- lambda_1 = 1 and the leading eigenvector, both from closed form --------------
    n, adjacency = planted_two_cliques()
    weights = degrees(adjacency)
    values, vectors, iters, residuals = orthogonal_iteration(adjacency, weights, 2, 20240589)
    assert abs(values[0] - 1.0) <= 1e-12, values[0]
    root_degree = [math.sqrt(d) for d in weights]
    length = norm(root_degree)
    closed_form = [value / length for value in root_degree]
    leading = vectors[0]
    sign = 1.0 if dot(leading, closed_form) >= 0 else -1.0
    assert max(abs(sign * leading[i] - closed_form[i]) for i in range(n)) <= 1e-9, \
        "the leading eigenvector is sqrt(d)/||sqrt(d)||"
    for i, residual in enumerate(residuals):
        assert residual <= 1e-9, (i, residual)
        image = normalized_matvec(adjacency, weights, vectors[i])
        assert abs(dot(vectors[i], image) - values[i]) <= 1e-12, "Rayleigh quotient"
        assert abs(norm(vectors[i]) - 1.0) <= 1e-12
    for i in range(2):
        for j in range(2):
            expected = 1.0 if i == j else 0.0
            assert abs(dot(vectors[i], vectors[j]) - expected) <= 1e-12, "orthonormal basis"
    assert iters <= 1000, "the iteration cap is 1000"

    # --- agreement with a dense decomposition (independent eigensolver) ---------------
    dense_values, dense_vectors = dense_spectrum(adjacency, weights)
    for i in range(2):
        assert abs(values[i] - dense_values[i]) <= 1e-9, (values[i], dense_values[i])
    # Eigenvalues of the normalized adjacency lie in [-1, 1]; the second one separates the
    # communities (close to 1 for a single bridge) and there is a spectral gap below it.
    assert -1.0 - 1e-9 <= values[1] <= 1.0 + 1e-9
    assert 0.9 <= values[1] < 1.0, values[1]
    assert dense_values[1] > dense_values[2] + 1e-6, "spectral gap"
    dense_projection = [sum(dense_vectors[i][j] * leading[j] for j in range(n)) for i in range(3)]
    assert abs(dense_projection[0]) >= 1.0 - 1e-9, "the leading dense vector is the same"
    assert abs(dense_projection[1]) <= 1e-9, "lambda_2 vector is orthogonal to lambda_1"

    # --- clustering matches the planted communities -----------------------------------
    labels, fiedler = fiedler_split(adjacency, weights, vectors[1])
    assert same_partition(labels, [0] * 6 + [1] * 6), labels
    assert labels.count(0) == 6 and labels.count(1) == 6
    swapped, _ = fiedler_split(adjacency, weights, [-value for value in vectors[1]])
    assert same_partition(swapped, labels), "the split is invariant up to relabelling"
    assert all(swapped[i] != labels[i] for i in range(n)), "flipping the sign relabels"
    # the Fiedler vector straddles zero, which is what makes the sign test legitimate
    assert min(fiedler) < 0.0 < max(fiedler)

    # --- three communities: multiple vectors plus k-means -----------------------------
    n3, adjacency3 = planted_three_cliques()
    weights3 = degrees(adjacency3)
    values3, vectors3, _, residuals3 = orthogonal_iteration(adjacency3, weights3, 3, 777)
    assert all(residual <= 1e-9 for residual in residuals3), residuals3
    dense3, _ = dense_spectrum(adjacency3, weights3)
    for i in range(3):
        assert abs(values3[i] - dense3[i]) <= 1e-9, (i, values3[i], dense3[i])
    assert abs(values3[0] - 1.0) <= 1e-12 and 0.9 <= values3[1] < 1.0
    rows = normalized_rows(vectors3, weights3, n3)
    labels3 = kmeans(rows, 3, 4242)
    planted3 = [0] * 5 + [1] * 5 + [2] * 5
    assert all(labels3[i] == labels3[planted3.index(planted3[i])] for i in range(n3)), labels3
    assert len(set(labels3)) == 3, labels3
    # The rows of U are not near canonical basis vectors (the rotation inside the spectral
    # subspace is arbitrary), but they do form tight, widely separated groups -- measured here:
    # the largest within-community distance is ~0.23 while the smallest cross-community distance
    # is ~1.2, which is the property the k-means step actually exploits.
    within = 0.0
    between = 2.0
    for i in range(n3):
        for j in range(i + 1, n3):
            distance = norm([a - b for a, b in zip(rows[i], rows[j])])
            if planted3[i] == planted3[j]:
                within = max(within, distance)
            else:
                between = min(between, distance)
    assert within <= 0.25, within
    assert between >= 1.0, between
    assert between > 4.0 * within

    # --- convergence rate: the leading vector's error decays like lambda_2 -------------
    # lambda_1 = 1, so the error of plain power iteration on the leading eigenvector decays as
    # (lambda_2 / lambda_1)^t = lambda_2^t; the measured ratio and the measured iteration count
    # are both compared with the eigenvalue obtained from the dense decomposition.
    length = norm(root_degree)
    root = [value / length for value in root_degree]
    state = 99
    probe = []
    for _ in range(n):
        state, pair = gauss_pair(state)
        probe.append(pair[0])
    probe = [value / norm(probe) for value in probe]
    errors = []
    iterations_to_1e_9 = None
    for step in range(700):
        probe = normalized_matvec(adjacency, weights, probe)
        probe = [value / norm(probe) for value in probe]
        sign = 1.0 if dot(probe, root) >= 0 else -1.0
        error = norm([sign * probe[i] - root[i] for i in range(n)])
        errors.append(error)
        if error <= 1e-9 and iterations_to_1e_9 is None:
            iterations_to_1e_9 = step + 1
    ratio = errors[59] / errors[58]
    assert abs(ratio - values[1]) <= 0.02 * values[1], (ratio, values[1])
    assert ratio < 1.0, "the error does shrink"
    predicted = math.log(1e-9) / math.log(values[1])
    assert iterations_to_1e_9 is not None
    assert 0.8 * predicted <= iterations_to_1e_9 <= 1.2 * predicted, \
        (iterations_to_1e_9, predicted)
    # A single float power iteration is far too slow for large graphs, which is why the file
    # iterates k orthonormal vectors at once; the subspace tolerance of 1e-13 costs ~500 steps
    # for lambda_2 = 0.949 (the observed count is asserted above the rate lower bound).
    assert iters >= 0.5 * math.log(1e-13) / math.log(values[1]), iters
    assert iters <= 3.0 * math.log(1e-13) / math.log(values[1]) + 10, iters

    # --- documented rate caveat: well-separated communities converge more slowly ------
    n_weak, weak = planted_one_community()
    weak_weights = degrees(weak)
    weak_values, weak_vectors, weak_iters, _ = orthogonal_iteration(weak, weak_weights, 2, 20240589)
    weak_dense, _ = dense_spectrum(weak, weak_weights)
    assert weak_dense[1] < dense_values[1] - 1e-3, \
        "the well-separated instance has the larger lambda_2, hence the slower iteration"
    assert iters > weak_iters, (iters, weak_iters)
    assert weak_iters >= 0.5 * math.log(1e-13) / math.log(weak_dense[1]), weak_iters
    assert abs(weak_values[0] - 1.0) <= 1e-12
    weak_labels, _ = fiedler_split(weak, weak_weights, weak_vectors[1])
    assert same_partition(weak_labels, labels), "both graphs split along the same communities"

    # --- determinism ------------------------------------------------------------------
    again, again_vectors, _, _ = orthogonal_iteration(adjacency, weights, 2, 20240589)
    assert again == values and again_vectors == vectors, "identical on a second run"
    assert kmeans(rows, 3, 4242) == labels3


if __name__ == "__main__":
    main()
    print("PASS")
