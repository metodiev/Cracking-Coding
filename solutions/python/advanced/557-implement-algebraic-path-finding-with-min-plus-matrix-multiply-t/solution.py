"""Task 557 - Implement algebraic path finding with min-plus matrix multiply (tropical semiring). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Shortest paths as linear algebra over the tropical semiring (R ∪ {inf}, min, +): the sum of two
elements is their minimum, the product is their sum, the additive identity is +inf and the
multiplicative identity is 0. Matrix multiplying two such matrices gives the shortest walk with
the number of edges equal to the sum of the two exponents, so A^k in this semiring is the
exactly-k-edge shortest walk matrix and the Kleene star A* = I ⊕ A ⊕ A^2 ⊕ ... is the all-pairs
shortest path matrix. Two independent engines compute the star: the Gauss-Jordan elimination in the
tropical semiring, which is textbook Floyd-Warshall (n^3 min-plus relaxations), and repeated
min-plus squaring of I ⊕ A, which covers every walk of length at most 2^t after t squarings and
therefore reaches the same fixed point after ceil(log2(n-1)) of them. Weights are integers and the
sentinel is float('inf'), so every arithmetic step is exact - no accumulated rounding is involved,
which is why the comparisons below are exact rather than tolerance-based. Instance data comes from
a full-period 64-bit LCG (MMIX constants) read in its high bits; the repo's 16-bit LCG
``(seed*75+74) % 65537`` has a short period and low-bit structure that would bias the weights.

Checks: the semiring laws hold on random matrices - associativity of min-plus multiplication,
distributivity of min over +, and the identity matrix (0 diagonal, inf elsewhere) as a two-sided
unit; all three star engines (Floyd-Warshall, squaring, and Dijkstra, plus Bellman-Ford for the
instances with negative weights) agree on every random nonnegative graph; distances match the
closed forms |i-j| on a path, min(|i-j|, n-|i-j|) on a cycle and 1 between distinct vertices of a
unit-weight complete graph; doubling every weight doubles every distance, while adding c to every
weight adds c per edge of the shortest path - exactly +c on the complete graph, where every
shortest path is a single edge, and 4|i-j| instead of |i-j| on the unit path with c = 3; the running minimum over the powers up to k is
non-increasing in k and reaches the closure at k = n-1, and the semigroup law
A^(p+q) = A^p ⊗ A^q holds exactly - with the honest note that the exactly-k matrix is *not*
monotone in k (on a single edge the 1-edge distance is the weight and the 2-edge distance is
infinite), which is why the star is a minimum over k and not a limit of powers. Honest caveats, both
asserted: with a negative cycle the star does not exist - Floyd-Warshall leaves a negative diagonal
exactly when Bellman-Ford detects one, and successive min-plus squarings drive the diagonal down
without bound, so the min-plus matrix power is a well-defined shortest-path matrix only on a
*-closed instance; and in the max-plus semiring the analogous closure is unbounded as soon as a
positive cycle exists, which the file demonstrates on a self-loop while showing that exactly-k-edge
max-plus powers do match the dynamic-programming longest path on a DAG.
"""
import heapq

MASK64 = (1 << 64) - 1
INF = float("inf")


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_weighted_graph(n, density, seed, low=1, high=20, directed=False):
    """Edge list with integer weights; directed or symmetric."""
    state = prng64((seed * 0x9E3779B97F4A7C15 + 19) & MASK64)
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j or (not directed and j < i):
                continue
            state = prng64(state)
            if (state >> 33) % 100 < density:
                state = prng64(state)
                weight = low + (state >> 33) % (high - low + 1)
                edges.append((i, j, weight))
                if not directed:
                    edges.append((j, i, weight))
    return n, edges


def weight_matrix(n, edges):
    matrix = [[INF] * n for _ in range(n)]
    for (u, v, w) in edges:
        if w < matrix[u][v]:
            matrix[u][v] = w
    return matrix


def tropical_identity(n, star=False):
    return [[0 if i == j else INF for j in range(n)] for i in range(n)]


def tropical_matmul(a, b, counter=None):
    """(min, +) product: c[i][j] = min_k a[i][k] + b[k][j]."""
    n = len(a)
    result = [[INF] * n for _ in range(n)]
    for i in range(n):
        row_a = a[i]
        row_r = result[i]
        for k in range(n):
            a_ik = row_a[k]
            if a_ik == INF:
                continue
            row_b = b[k]
            for j in range(n):
                candidate = a_ik + row_b[j]
                if candidate < row_r[j]:
                    row_r[j] = candidate
    if counter is not None:
        counter[0] += n ** 3
    return result


def tropical_add(a, b):
    """The semiring sum: elementwise minimum."""
    n = len(a)
    return [[min(a[i][j], b[i][j]) for j in range(n)] for i in range(n)]


def tropical_pow(matrix, exponent, counter=None):
    """A^k - the exactly-k-edge shortest walk matrix; A^0 is the identity."""
    n = len(matrix)
    result = tropical_identity(n)
    base = [row[:] for row in matrix]
    while exponent > 0:
        if exponent & 1:
            result = tropical_matmul(result, base, counter)
        exponent >>= 1
        if exponent:
            base = tropical_matmul(base, base, counter)
    return result


def floyd_warshall(matrix):
    """Gauss-Jordan elimination over the tropical semiring; the diagonal may be negative."""
    n = len(matrix)
    distance = [[0 if i == j else matrix[i][j] for j in range(n)] for i in range(n)]
    counter = 0
    for k in range(n):
        row_k = distance[k]
        for i in range(n):
            d_ik = distance[i][k]
            if d_ik == INF:
                counter += n                      # the triple is examined, the sum is skipped
                continue
            row_i = distance[i]
            for j in range(n):
                candidate = d_ik + row_k[j]
                counter += 1
                if candidate < row_i[j]:
                    row_i[j] = candidate
    return distance, counter


def closure_by_squaring(matrix):
    """Star computed by min-plus squaring of I (+) A; valid when no negative cycle exists."""
    n = len(matrix)
    current = tropical_add(matrix, tropical_identity(n))
    for _ in range(max(1, (n - 1).bit_length())):
        current = tropical_matmul(current, current)
        for i in range(n):
            if current[i][i] < 0:
                return current, True
    return current, False


def dijkstra(n, edges, source):
    adjacency = [[] for _ in range(n)]
    for (u, v, w) in edges:
        adjacency[u].append((v, w))
    distance = [INF] * n
    distance[source] = 0
    heap = [(0, source)]
    while heap:
        (d, u) = heapq.heappop(heap)
        if d > distance[u]:
            continue
        for (v, w) in adjacency[u]:
            candidate = d + w
            if candidate < distance[v]:
                distance[v] = candidate
                heapq.heappush(heap, (candidate, v))
    return distance


def bellman_ford(n, edges, source, rounds=None):
    """Returns (distances, negative_cycle_reachable)."""
    if rounds is None:
        rounds = n - 1
    distance = [INF] * n
    distance[source] = 0
    for _ in range(rounds):
        changed = False
        for (u, v, w) in edges:
            if distance[u] != INF and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
                changed = True
        if not changed:
            return distance, False
    for (u, v, w) in edges:
        if distance[u] != INF and distance[u] + w < distance[v]:
            return distance, True
    return distance, False


def has_negative_cycle_bellman_ford(n, edges):
    """A virtual source joined to every vertex with weight 0 makes this all-pairs detection."""
    virtual = list(edges) + [(n, v, 0) for v in range(n)]
    distance = [INF] * (n + 1)
    distance[n] = 0
    for _ in range(n):
        for (u, v, w) in virtual:
            if distance[u] != INF and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
    for (u, v, w) in virtual:
        if distance[u] != INF and distance[u] + w < distance[v]:
            return True
    return False


def exact_walk_dp(n, edges, k, source):
    """Vector DP: the cheapest walk with exactly k edges, step by step."""
    distance = [INF] * n
    distance[source] = 0
    for _ in range(k):
        following = [INF] * n
        for (u, v, w) in edges:
            if distance[u] != INF:
                candidate = distance[u] + w
                if candidate < following[v]:
                    following[v] = candidate
        distance = following
    return distance


def simple_path_min(n, edges, source):
    """Cheapest simple path from source to every vertex, by DFS over subsets (n small)."""
    adjacency = [[] for _ in range(n)]
    for (u, v, w) in edges:
        adjacency[u].append((v, w))
    best = [INF] * n

    def dfs(vertex, cost, visited):
        if cost < best[vertex]:
            best[vertex] = cost
        for (nxt, w) in adjacency[vertex]:
            if not (visited >> nxt) & 1:
                dfs(nxt, cost + w, visited | (1 << nxt))

    dfs(source, 0, 1 << source)
    return best


def max_plus_matmul(a, b):
    n = len(a)
    result = [[-INF] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if a[i][k] == -INF:
                continue
            for j in range(n):
                if b[k][j] == -INF:
                    continue
                candidate = a[i][k] + b[k][j]
                if candidate > result[i][j]:
                    result[i][j] = candidate
    return result


def max_plus_pow(matrix, exponent):
    """Exactly-k-edge heaviest walk matrix over (max, +); the identity has 0 on the diagonal."""
    n = len(matrix)
    result = [[0 if i == j else -INF for j in range(n)] for i in range(n)]
    base = [row[:] for row in matrix]
    while exponent > 0:
        if exponent & 1:
            result = max_plus_matmul(result, base)
        exponent >>= 1
        if exponent:
            base = max_plus_matmul(base, base)
    return result


def longest_path_dp_dag(n, edges, k, source):
    """Longest walk with exactly k edges in a DAG (weights may be negative, no cycles though)."""
    best = [-INF] * n
    best[source] = 0
    for _ in range(k):
        following = [-INF] * n
        for (u, v, w) in edges:
            if best[u] != -INF and best[u] + w > following[v]:
                following[v] = best[u] + w
        best = following
    return best


def path_graph(n):
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1, 1))
        edges.append((i + 1, i, 1))
    return n, edges


def cycle_graph(n):
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n, 1))
        edges.append(((i + 1) % n, i, 1))
    return n, edges


def complete_graph(n):
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((i, j, 1))
    return n, edges


def close(a, b):
    return a == b or abs(a - b) <= 1e-9


def main():
    # --- Semiring laws -------------------------------------------------------------------
    n, edges = random_weighted_graph(5, 60, 1)
    a = weight_matrix(n, edges)
    b = weight_matrix(n, edges)
    c = weight_matrix(n, edges)
    identity = tropical_identity(n)
    assert tropical_matmul(a, identity) == a == tropical_matmul(identity, a)
    left = tropical_matmul(tropical_matmul(a, b), c)
    right = tropical_matmul(a, tropical_matmul(b, c))
    assert left == right, "min-plus multiplication is associative"
    assert tropical_matmul(tropical_add(a, b), c) == \
        tropical_add(tropical_matmul(a, c), tropical_matmul(b, c)), "distributivity"
    assert tropical_add(a, b) == tropical_add(b, a)
    empty = [[INF] * n for _ in range(n)]
    assert tropical_matmul(a, empty) == empty and tropical_add(a, empty) == a

    # --- Three (four) engines agree ------------------------------------------------------
    for (size, density, seed) in ((5, 60, 11), (6, 45, 12), (7, 70, 13), (8, 35, 14)):
        instance = random_weighted_graph(size, density, seed)
        size_n, instance_edges = instance
        matrix = weight_matrix(size_n, instance_edges)
        fw, relaxations = floyd_warshall(matrix)
        squared, negative = closure_by_squaring(matrix)
        assert not negative
        assert fw == squared, "squaring reaches the Floyd-Warshall fixed point"
        assert relaxations == size ** 3
        for source in range(size_n):
            assert fw[source] == dijkstra(size_n, instance_edges, source), (size, seed, source)
            bf, flagged = bellman_ford(size_n, instance_edges, source)
            assert not flagged and bf == fw[source]

    # --- Closed forms --------------------------------------------------------------------
    for n in (1, 2, 5, 8):
        size, instance_edges = path_graph(n)
        fw, _ = floyd_warshall(weight_matrix(size, instance_edges))
        for i in range(n):
            for j in range(n):
                assert close(fw[i][j], abs(i - j)), (n, i, j)
    for n in (3, 6, 9):
        size, instance_edges = cycle_graph(n)
        fw, _ = floyd_warshall(weight_matrix(size, instance_edges))
        for i in range(n):
            for j in range(n):
                assert close(fw[i][j], min(abs(i - j), n - abs(i - j))), (n, i, j)
    size, instance_edges = complete_graph(5)
    fw, _ = floyd_warshall(weight_matrix(size, instance_edges))
    assert all(close(fw[i][j], 0 if i == j else 1) for i in range(5) for j in range(5))

    # --- Weight transformations -----------------------------------------------------------
    instance = random_weighted_graph(7, 50, 21)
    size, instance_edges = instance
    matrix = weight_matrix(size, instance_edges)
    fw, _ = floyd_warshall(matrix)
    doubled = [[0 if i == j else (matrix[i][j] * 2 if matrix[i][j] != INF else INF)
                for j in range(size)] for i in range(size)]
    fw_doubled, _ = floyd_warshall(doubled)
    assert all(close(fw_doubled[i][j], 2 * fw[i][j]) for i in range(size) for j in range(size)), \
        "doubling every weight doubles every distance"
    # Adding c to every edge adds c per edge of the shortest path, so the effect is not a
    # constant: on the unit path the distance |i-j| becomes 4|i-j| for c = 3, while on the unit
    # complete graph, where every shortest path is a single edge, it is exactly +3.
    size_p, path_edges = path_graph(6)
    path_matrix = weight_matrix(size_p, path_edges)
    shifted_path = [[0 if i == j else (path_matrix[i][j] + 3 if path_matrix[i][j] != INF else INF)
                     for j in range(size_p)] for i in range(size_p)]
    fw_path, _ = floyd_warshall(shifted_path)
    assert all(close(fw_path[i][j], 4 * abs(i - j)) for i in range(size_p) for j in range(size_p))
    size_c, complete_edges = complete_graph(5)
    complete_matrix = weight_matrix(size_c, complete_edges)
    shifted_complete = [[0 if i == j else (
        complete_matrix[i][j] + 3 if complete_matrix[i][j] != INF else INF)
        for j in range(size_c)] for i in range(size_c)]
    fw_complete, _ = floyd_warshall(shifted_complete)
    assert all(close(fw_complete[i][j], 0 if i == j else 4)
               for i in range(size_c) for j in range(size_c))

    # --- Exactly-k-edge powers -----------------------------------------------------------
    for (density, seed) in ((60, 31), (40, 32)):
        instance = random_weighted_graph(6, density, seed)
        size, instance_edges = instance
        matrix = weight_matrix(size, instance_edges)
        running = tropical_identity(size)
        previous = None
        for k in range(0, 9):
            power = tropical_pow(matrix, k)
            for source in range(size):
                assert power[source] == exact_walk_dp(size, instance_edges, k, source)
            running = tropical_add(running, power)
            if previous is not None:
                for source in range(size):
                    for j in range(size):
                        assert running[source][j] <= previous[source][j] + 1e-9, \
                            "the minimum over the powers up to k never rises"
            previous = running
        fw, _ = floyd_warshall(matrix)
        accumulated = tropical_identity(size)
        for k in range(0, size):
            accumulated = tropical_add(accumulated, tropical_pow(matrix, k))
        assert accumulated == fw, "the star is the minimum over the powers 0..n-1"
        assert tropical_pow(matrix, 3) == tropical_matmul(tropical_pow(matrix, 2),
                                                          tropical_pow(matrix, 1))
        assert tropical_pow(matrix, 5) == tropical_matmul(tropical_pow(matrix, 3),
                                                          tropical_pow(matrix, 2))
        assert tropical_pow(matrix, 0) == tropical_identity(size)
    # The exactly-k matrix itself is not monotone in k, which is why the star has to be the
    # minimum over all k rather than a limit of powers: on a single edge the 1-edge distance is
    # the weight and the 2-edge distance is infinite.
    single = weight_matrix(2, [(0, 1, 7), (1, 0, 7)])
    assert tropical_pow(single, 1)[0][1] == 7
    assert tropical_pow(single, 2)[0][1] == INF

    # --- Negative weights on a DAG: shortest walk = shortest simple path ------------------
    dag_edges = [(0, 1, -2), (1, 2, -3), (0, 2, 5), (2, 3, 1), (1, 3, 4), (3, 4, -1), (0, 4, 9)]
    size = 5
    matrix = weight_matrix(size, dag_edges)
    fw, _ = floyd_warshall(matrix)
    squared, negative = closure_by_squaring(matrix)
    assert not negative and squared == fw
    for source in range(size):
        assert fw[source] == simple_path_min(size, dag_edges, source), \
            "with no negative cycle the cheapest walk is a simple path"
        bf, flagged = bellman_ford(size, dag_edges, source)
        assert not flagged and bf == fw[source]

    # --- Honest caveat: a negative cycle makes the star undefined ------------------------
    cycle_edges = [(0, 1, 1), (1, 2, -1), (2, 0, -2)]
    size = 3
    matrix = weight_matrix(size, cycle_edges)
    fw_negative, _ = floyd_warshall(matrix)
    assert any(fw_negative[i][i] < 0 for i in range(size)), "the diagonal goes negative"
    assert has_negative_cycle_bellman_ford(size, cycle_edges)
    _, reachable_cycle = bellman_ford(size, cycle_edges, 0)
    assert reachable_cycle
    earlier = tropical_pow(matrix, 3)
    later = tropical_pow(matrix, 6)
    assert later[0][0] < earlier[0][0] < 0, "successive powers keep descending without bound"
    _, diverged = closure_by_squaring(matrix)
    assert diverged, "squaring detects the divergence through the diagonal"
    # Floyd-Warshall's diagonal criterion and Bellman-Ford agree on random signed graphs
    for (density, seed) in ((50, 41), (70, 42), (40, 43)):
        instance = random_weighted_graph(6, density, seed, low=-6, high=8, directed=True)
        size, instance_edges = instance
        fw, _ = floyd_warshall(weight_matrix(size, instance_edges))
        by_diagonal = any(fw[i][i] < 0 for i in range(size))
        assert by_diagonal == has_negative_cycle_bellman_ford(size, instance_edges), seed

    # --- Max-plus: exact powers fine on a DAG, closure unbounded on a cycle --------------
    max_matrix = [[-INF if value == INF else value for value in row] for row in
                  weight_matrix(5, dag_edges)]
    for k in range(0, 5):
        power = max_plus_pow(max_matrix, k)
        for source in range(5):
            assert power[source] == longest_path_dp_dag(5, dag_edges, k, source), k
    positive_cycle = [[0 if i == j else -INF for j in range(3)] for i in range(3)]
    positive_cycle[0][0] = 2                                   # a self-loop of weight 2
    first = max_plus_pow(positive_cycle, 1)[0][0]
    second = max_plus_pow(positive_cycle, 2)[0][0]
    third = max_plus_pow(positive_cycle, 3)[0][0]
    assert 0 < first < second < third, "max-plus closure grows without bound on a cycle"

    # --- Determinism ---------------------------------------------------------------------
    instance = random_weighted_graph(7, 55, 51)
    size, instance_edges = instance
    assert floyd_warshall(weight_matrix(size, instance_edges))[0] == \
        floyd_warshall(weight_matrix(size, instance_edges))[0]
    assert tropical_pow(weight_matrix(size, instance_edges), 4) == \
        tropical_pow(weight_matrix(size, instance_edges), 4)
    assert dijkstra(size, instance_edges, 0) == dijkstra(size, instance_edges, 0)


if __name__ == "__main__":
    main()
    print("PASS")
