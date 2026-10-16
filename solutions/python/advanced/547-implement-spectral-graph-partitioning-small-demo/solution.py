"""Task 547 - Implement spectral graph partitioning small demo. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Spectral bisection of a small graph: build the combinatorial Laplacian L = D - A, compute its full
spectrum with cyclic Jacobi rotations (sweeps until the off-diagonal Frobenius norm drops below
1e-13, capped at 200 sweeps), and split the vertices by the Fiedler vector - the eigenvector of the
second-smallest eigenvalue. The split threshold is the mean of the Fiedler vector, with the sign
fixed so that vertex 0 sits at or above the threshold, which makes the labelling deterministic; a
second variant splits the sorted Fiedler values at the median to produce an exactly balanced
bisection, and a sweep variant scans every prefix of the sorted vector and keeps the one with the
smallest conductance (the standard spectral sweep argument).

Everything is compared against exhaustive searches on small graphs: the minimum cut over all
2^(n-1) bipartitions, the minimum bisection over all equal-sized splits, and the minimum
conductance via the brute-force scan of the same splits. The theory-backed invariants are checked
too: L is symmetric positive semi-definite with an all-zero eigenvalue pair for every connected
component (so a disconnected graph has Fiedler value 0 and the Fiedler split returns the
components), the eigen-pair residual ||L v - lambda v|| and orthogonality of v to the constant are
below 1e-9, and the closed forms lambda2 = 2 - 2 cos(pi/n) for the path and
lambda2 = 2 - 2 cos(2 pi/n) for the cycle are reproduced, as is lambda2 = n for the complete graph
(with every non-zero eigenvalue equal to n) and the minimum bisection 4 of the 4x4 grid.

Checks: the two triangles joined by a single bridge (Fiedler split = the triangles, cut 1, equal to
the exhaustive minimum cut, and its conductance 1/7 equal to the exhaustive minimum); the
disconnected graph (two zero eigenvalues, and the balanced split recovering the components that an
independent traversal finds - the mean-threshold split degenerates there because the zero
eigenspace is two-dimensional and Jacobi returns an arbitrary basis of it); the path (monotone
Fiedler vector, closed form 2 - 2 cos(pi/n), median split at the middle); the cycle (two arcs,
2 - 2 cos(2 pi/n)); K5 (spectrum {0, 5, 5, 5, 5}, every bisection cutting 6 edges); and the 4x4
grid, whose whole spectrum must equal the pairwise sums of the P4 spectrum and whose optimal
bisection cuts 4 - a value the Fiedler thresholds and the sweep do *not* reach (their staircase
prefixes cut 6), which the file asserts explicitly rather than pretending otherwise, together with
the factor-three conductance bound and the identity cut(S) = degrees(S) - 2 * inner(S).
"""
import math

TOL = 1e-13


def graph_from_edges(n, edges):
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        assert u != v, "no self loops in the demo graphs"
        adj[u].append(v)
        adj[v].append(u)
    for v in range(n):
        adj[v].sort()
    return adj


def laplacian(n, adj):
    matrix = [[0.0] * n for _ in range(n)]
    for v in range(n):
        matrix[v][v] = float(len(adj[v]))
        for w in adj[v]:
            matrix[v][w] = -1.0
    return matrix


def jacobi_eigh(matrix, max_sweeps=200, tol=TOL):
    """Cyclic Jacobi eigen-decomposition; returns (eigenvalues, eigenvectors as columns, sweeps)."""
    n = len(matrix)
    a = [list(row) for row in matrix]
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


def spectrum(n, adj):
    return jacobi_eigh(laplacian(n, adj))


def fiedler(n, adj):
    """Returns (lambda2, Fiedler vector, residual, orthogonality error, sweeps, all eigenvalues)."""
    values, vectors, sweeps = spectrum(n, adj)
    order = sorted(range(n), key=lambda i: (values[i], i))
    index = order[1]
    lam2 = values[index]
    vector = [vectors[i][index] for i in range(n)]
    matrix = laplacian(n, adj)
    residual = math.sqrt(sum((sum(matrix[i][j] * vector[j] for j in range(n))
                              - lam2 * vector[i]) ** 2 for i in range(n)))
    orthogonality = abs(sum(vector))
    return lam2, vector, residual, orthogonality, sweeps, [values[i] for i in range(n)]


def sign_partition(vector):
    """Split by the mean of the Fiedler vector, with the sign pinned by vertex 0."""
    n = len(vector)
    oriented = list(vector)
    mean = sum(oriented) / n
    if oriented[0] < mean:
        oriented = [-value for value in oriented]
    return [0 if oriented[i] >= mean else 1 for i in range(n)]


def median_partition(vector):
    """Balanced bisection: the smallest half of the Fiedler values against the largest half."""
    n = len(vector)
    order = sorted(range(n), key=lambda i: (vector[i], i))
    half = n // 2
    labels = [0] * n
    for position in order[half:]:
        labels[position] = 1
    return labels


def cut_size(adj, labels):
    total = 0
    for u in range(len(adj)):
        for v in adj[u]:
            if u < v and labels[u] != labels[v]:
                total += 1
    return total


def partition_of(labels):
    """Order-insensitive view of a labelling: the set of sides."""
    return frozenset(frozenset(v for v in range(len(labels)) if labels[v] == value)
                     for value in set(labels))


def components(adj):
    """Connected components by explicit traversal - an independent reference for the split."""
    n = len(adj)
    seen = [False] * n
    out = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        cell = []
        while stack:
            v = stack.pop()
            cell.append(v)
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        out.append(frozenset(cell))
    return frozenset(out)


def degrees(adj):
    return [len(adj[v]) for v in range(len(adj))]


def cut_by_identity(adj, labels):
    """Independent cut computation: sum of degrees inside one side minus twice the inner edges."""
    side = [v for v in range(len(adj)) if labels[v] == 1]
    deg = degrees(adj)
    total = sum(deg[v] for v in side)
    inner = 0
    for u in side:
        for v in adj[u]:
            if u < v and labels[v] == 1:
                inner += 1
    return total - 2 * inner


def volume(adj, labels, value):
    deg = degrees(adj)
    return sum(deg[v] for v in range(len(adj)) if labels[v] == value)


def conductance(adj, labels):
    cut = cut_size(adj, labels)
    if cut == 0:
        return 0.0
    left = volume(adj, labels, 0)
    right = volume(adj, labels, 1)
    if left == 0 or right == 0:
        return float("inf")
    return cut / min(left, right)


def brute_force_min_cut(n, adj):
    best = None
    best_labels = None
    for mask in range(1 << (n - 1)):
        labels = [0] * n
        for index in range(n - 1):
            labels[index + 1] = (mask >> index) & 1
        if all(value == labels[0] for value in labels):
            continue
        value = cut_size(adj, labels)
        if best is None or value < best:
            best = value
            best_labels = labels
    return best, best_labels


def brute_force_min_bisection(n, adj):
    """Minimum cut over all splits with exactly floor(n/2) vertices on one side."""
    best = None
    best_labels = None
    half = n // 2
    for mask in range(1 << n):
        labels = [(mask >> i) & 1 for i in range(n)]
        if sum(labels) != half:
            continue
        value = cut_size(adj, labels)
        if best is None or value < best:
            best = value
            best_labels = labels
    return best, best_labels


def brute_force_min_conductance(n, adj):
    best = None
    for mask in range(1, (1 << n) - 1):
        labels = [(mask >> i) & 1 for i in range(n)]
        value = conductance(adj, labels)
        if best is None or value < best:
            best = value
    return best


def sweep_best_prefix(adj, vector):
    """The best conductance among all prefixes of the Fiedler order (the sweep heuristic)."""
    n = len(adj)
    order = sorted(range(n), key=lambda i: (vector[i], i))
    best = None
    best_labels = None
    for size in range(1, n):
        labels = [1] * n
        for position in order[:size]:
            labels[position] = 0
        value = conductance(adj, labels)
        if best is None or value < best:
            best = value
            best_labels = labels
    return best, best_labels


def two_cliques(k, bridge=True):
    n = 2 * k
    edges = [(i, j) for i in range(k) for j in range(i + 1, k)]
    edges += [(k + i, k + j) for i in range(k) for j in range(i + 1, k)]
    if bridge:
        edges.append((0, k))
    return n, edges


def path(n):
    return n, [(i, i + 1) for i in range(n - 1)]


def cycle(n):
    return n, [(i, (i + 1) % n) for i in range(n)]


def grid(rows, cols):
    n = rows * cols
    edges = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    return n, edges


def complete(n):
    return n, [(i, j) for i in range(n) for j in range(i + 1, n)]


def disconnected_cliques(k):
    n = 2 * k
    edges = [(i, j) for i in range(k) for j in range(i + 1, k)]
    edges += [(k + i, k + j) for i in range(k) for j in range(i + 1, k)]
    return n, edges


def main():
    # --- Two triangles plus a bridge ----------------------------------------------------
    n, edges = two_cliques(3)
    adj = graph_from_edges(n, edges)
    lam2, vector, residual, orthogonality, sweeps, values = fiedler(n, adj)
    assert residual <= 1e-9, "eigen-pair residual"
    assert orthogonality <= 1e-8, "the Fiedler vector is orthogonal to the constant"
    assert sweeps <= 200
    assert min(values) >= -1e-9, "L is positive semi-definite"
    assert lam2 > 1e-9, "a connected graph has a strictly positive Fiedler value"
    labels = sign_partition(vector)
    assert labels == [0, 0, 0, 1, 1, 1], "the bridge is the only edge between the triangles"
    assert cut_size(adj, labels) == 1
    assert cut_by_identity(adj, labels) == 1, "two independent cut computations"
    minimum, _ = brute_force_min_cut(n, adj)
    assert minimum == 1 and cut_size(adj, labels) == minimum
    assert volume(adj, labels, 0) == 7 and volume(adj, labels, 1) == 7, \
        "the bridge endpoint has degree 3, so each side has volume 2 + 2 + 3"
    assert abs(conductance(adj, labels) - 1.0 / 7.0) <= 1e-12, "cut 1 over the smaller volume 7"
    assert abs(brute_force_min_conductance(n, adj) - conductance(adj, labels)) <= 1e-12, \
        "the spectral split also minimises conductance here"
    sweep_value, sweep_labels = sweep_best_prefix(adj, vector)
    assert sweep_value <= conductance(adj, labels) + 1e-12, "the sweep finds something at least as good"

    # --- Disconnected graph: two zero eigenvalues ---------------------------------------
    n, edges = disconnected_cliques(3)
    adj = graph_from_edges(n, edges)
    values_sorted = sorted(spectrum(n, adj)[0])
    assert abs(values_sorted[0]) <= 1e-9 and abs(values_sorted[1]) <= 1e-9, \
        "one zero eigenvalue per connected component"
    assert values_sorted[2] > 1e-6
    lam2, vector, residual, orthogonality, _, _ = fiedler(n, adj)
    assert abs(lam2) <= 1e-9
    # The zero eigenvalue is two-fold here, so the eigenvector Jacobi returns is only *some*
    # vector of that eigenspace (it can be a single component's indicator) and the mean-threshold
    # split degenerates. The balanced split is the meaningful one, and it recovers the components.
    assert components(adj) == frozenset({frozenset({0, 1, 2}), frozenset({3, 4, 5})})
    labels = median_partition(vector)
    assert partition_of(labels) == components(adj), "the Fiedler order recovers the components"
    assert cut_size(adj, labels) == 0 and brute_force_min_cut(n, adj)[0] == 0
    degenerate = sign_partition(vector)
    assert min(cut_size(adj, degenerate), cut_size(adj, median_partition(vector))) == 0

    # --- Path: closed-form eigenvalue and a monotone Fiedler vector --------------------
    for size in (4, 5, 6, 7):
        n, edges = path(size)
        adj = graph_from_edges(n, edges)
        lam2, vector, residual, orthogonality, _, _ = fiedler(n, adj)
        closed = 2.0 - 2.0 * math.cos(math.pi / size)
        assert abs(lam2 - closed) <= 1e-9, "the path has lambda2 = 2 - 2 cos(pi/n)"
        assert residual <= 1e-9 and orthogonality <= 1e-8
        monotone = all(vector[i] <= vector[i + 1] + 1e-12 for i in range(size - 1)) or \
            all(vector[i] >= vector[i + 1] - 1e-12 for i in range(size - 1))
        assert monotone, "the Fiedler vector of a path is monotone"
        assert median_partition(vector) == [0] * (size // 2) + [1] * (size - size // 2), \
            "the balanced split cuts the path in the middle"
    n, edges = path(6)
    adj = graph_from_edges(n, edges)
    lam2, vector, _, _, _, _ = fiedler(n, adj)
    assert cut_size(adj, median_partition(vector)) == 1
    assert brute_force_min_bisection(n, adj)[0] == 1

    # --- Cycle: closed form and two arcs ------------------------------------------------
    for size in (4, 6, 8):
        n, edges = cycle(size)
        adj = graph_from_edges(n, edges)
        lam2, vector, residual, _, _, _ = fiedler(n, adj)
        closed = 2.0 - 2.0 * math.cos(2.0 * math.pi / size)
        assert abs(lam2 - closed) <= 1e-9, "the cycle has lambda2 = 2 - 2 cos(2 pi/n)"
        labels = median_partition(vector)
        assert cut_size(adj, labels) == 2, "an even cycle bisects into two arcs"
        assert residual <= 1e-9

    # --- Complete graph: every non-zero eigenvalue equals n ----------------------------
    n, edges = complete(5)
    adj = graph_from_edges(n, edges)
    values, _, _ = spectrum(n, adj)
    ordered = sorted(values)
    assert abs(ordered[0]) <= 1e-9
    assert all(abs(ordered[i] - 5.0) <= 1e-9 for i in range(1, 5)), "spectrum of K5 is {0, 5, 5, 5, 5}"
    lam2, vector, _, _, _, _ = fiedler(n, adj)
    assert abs(lam2 - 5.0) <= 1e-9
    balanced = median_partition(vector)
    assert cut_size(adj, balanced) == 6, "K5 bisection cuts 2 * 3 = 6 edges"
    assert brute_force_min_bisection(n, adj)[0] == 6

    # --- Grid: the spectrum is the pairwise sum of the path spectrum --------------------
    n, edges = grid(4, 4)
    adj = graph_from_edges(n, edges)
    values, _, _ = spectrum(n, adj)
    path_values = [2.0 - 2.0 * math.cos(math.pi * k / 4.0) for k in range(4)]
    expected = sorted(a + b for a in path_values for b in path_values)
    got = sorted(values)
    assert all(abs(got[i] - expected[i]) <= 1e-9 for i in range(16)), \
        "the Laplacian of P4 x P4 has the pairwise sums of the P4 spectrum"
    lam2, vector, residual, orthogonality, _, _ = fiedler(n, adj)
    assert abs(lam2 - (2.0 - 2.0 * math.cos(math.pi / 4.0))) <= 1e-9, \
        "the smallest non-zero grid eigenvalue is lambda2 of P4, and it is two-fold"
    assert residual <= 1e-9 and orthogonality <= 1e-8
    assert brute_force_min_bisection(n, adj)[0] == 4, "the optimal bisection cuts four edges"
    assert brute_force_min_cut(n, adj)[0] == 2, "the minimum cut just peels off a corner"
    minimum_conductance = brute_force_min_conductance(n, adj)
    assert abs(minimum_conductance - 4.0 / 24.0) <= 1e-12, \
        "the best conductance splits the grid into two rows of volume 24 with a cut of 4"
    # Thresholding the Fiedler vector on a grid is *not* optimal - the Fiedler order runs
    # diagonally, so its prefixes are staircases that cut six edges rather than four. Both
    # threshold rules and the sweep are still within a small factor of the optimum.
    for labels in (sign_partition(vector), median_partition(vector)):
        assert cut_size(adj, labels) == 6
        assert cut_by_identity(adj, labels) == 6
        assert cut_size(adj, labels) >= brute_force_min_bisection(n, adj)[0]
    sweep_value, sweep_labels = sweep_best_prefix(adj, vector)
    assert abs(sweep_value - 0.25) <= 1e-12, "six cut edges over a volume of 24"
    assert cut_size(adj, sweep_labels) == 6
    assert sweep_value >= minimum_conductance - 1e-12, "no heuristic can beat the exhaustive best"
    assert sweep_value <= 3.0 * minimum_conductance + 1e-12, \
        "and the sweep stays within a factor three of it"

    # --- Determinism and symmetry -------------------------------------------------------
    n, edges = two_cliques(4)
    adj = graph_from_edges(n, edges)
    first = fiedler(n, adj)
    second = fiedler(n, adj)
    assert first[1] == second[1], "the eigensolver is deterministic"
    assert sign_partition(first[1]) == sign_partition(second[1])
    assert sorted(first[5]) == sorted(second[5])
    permutation = [0, 2, 1, 3, 5, 4, 7, 6]
    relabelled_edges = [(permutation[u], permutation[v]) for (u, v) in edges]
    labelled = graph_from_edges(n, relabelled_edges)
    third = fiedler(n, labelled)
    assert abs(third[0] - first[0]) <= 1e-9, "lambda2 does not depend on the labelling"
    assert cut_size(labelled, sign_partition(third[1])) == 1


if __name__ == "__main__":
    main()
    print("PASS")
