"""Task 555 - Implement dynamic programming for counting paths of length k in DAG (matrix exponentiation). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Counting paths with exactly k edges in a DAG three ways. (1) A vertex DP in the spirit of a
topological sweep: the vector v_t records, per ending vertex, the number of paths with t edges
(v_0 = a unit vector at the source, or all-ones when every source is allowed), and v_{t+1} is
obtained by summing v_t over the in-neighbours of each vertex - one pass per length, k passes.
(2) Matrix exponentiation: v_k is the source vector times A^k, computed by binary exponentiation
with exact integers, and the file checks the DP vector against the column sums of A^k and against
A^k applied to the start vector. (3) Brute-force DFS enumeration of walks of length k. The file
also runs the DP in topological order explicitly (Kahn's algorithm, smallest ready vertex first)
and asserts that the order is respected by every edge. Random DAGs are built by drawing each pair
i < j with a fixed density from a full-period 64-bit LCG (MMIX constants) read in its high bits;
the repo's 16-bit LCG ``(seed*75+74) % 65537`` has a short period and low-bit structure that would
correlate consecutive edge decisions.

The point the file makes precise: A^k counts *walks*, which equal paths only because a DAG has no
cycles. Checks: on random DAGs the three methods agree for every k and every starting vertex, the
number of paths summed over all lengths equals the brute-force count of all paths (2^n - 1 for the
complete DAG, which follows from sum_k C(n, k+1) = 2^n - 1), the complete DAG matches the binomial
closed forms C(n-1-i, k) per source and C(n, k+1) in total, A^k is the zero matrix for k >= n and
every v_k is zero for k > n-1, k = 0 gives the identity (n trivial paths, one per vertex), the
modular version of the fast power agrees with the exact one, and binary exponentiation performs at
most 2*ceil(log2 k) matrix multiplications against k-1 for repeated multiplication. Honest
counterexample, asserted: on the directed 3-cycle 0 -> 1 -> 2 -> 0 the matrix power says there are
3 closed walks of length 3 while there is no simple path of length 3 at all (and on the undirected
triangle it is 6 versus 0), so the matrix-power path count is only a path count on a DAG.
"""
MASK64 = (1 << 64) - 1
MOD = 1000000007


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_dag(n, density, seed):
    """Edges only from lower to higher index, so 0..n-1 is a topological order."""
    state = prng64((seed * 0x9E3779B97F4A7C15 + 13) & MASK64)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            state = prng64(state)
            if (state >> 33) % 100 < density:
                edges.append((i, j))
    return n, edges


def complete_dag(n):
    return n, [(i, j) for i in range(n) for j in range(i + 1, n)]


def incoming_lists(n, edges):
    incoming = [[] for _ in range(n)]
    for (u, v) in edges:
        incoming[v].append(u)
    return incoming


def topological_order(n, edges):
    """Kahn's algorithm with the smallest ready vertex first, for determinism."""
    incoming = incoming_lists(n, edges)
    indegree = [len(incoming[v]) for v in range(n)]
    outgoing = [[] for _ in range(n)]
    for (u, v) in edges:
        outgoing[u].append(v)
    ready = [v for v in range(n) if indegree[v] == 0]
    order = []
    while ready:
        ready.sort()
        u = ready.pop(0)
        order.append(u)
        for v in outgoing[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                ready.append(v)
    return order if len(order) == n else None


def path_vector_dp(n, edges, k, source=None):
    """(vector of path counts by ending vertex, total) for paths with exactly k edges."""
    incoming = incoming_lists(n, edges)
    vector = [1] * n if source is None else [1 if v == source else 0 for v in range(n)]
    for _ in range(k):
        following = [0] * n
        for v in range(n):
            total = 0
            for u in incoming[v]:
                total += vector[u]
            following[v] = total
        vector = following
    return vector, sum(vector)


def path_vector_dp_topological(n, edges, k, source=None):
    """The same DP, but relaxing vertices in an explicit topological order each round."""
    order = topological_order(n, edges)
    assert order is not None, "the instance must be acyclic"
    incoming = incoming_lists(n, edges)
    vector = [1] * n if source is None else [1 if v == source else 0 for v in range(n)]
    for _ in range(k):
        following = [0] * n
        for v in order:
            total = 0
            for u in incoming[v]:
                total += vector[u]
            following[v] = total
        vector = following
    return vector, sum(vector)


def adjacency_matrix(n, edges):
    matrix = [[0] * n for _ in range(n)]
    for (u, v) in edges:
        matrix[u][v] += 1
    return matrix


def matmul(a, b, mod=None):
    n = len(a)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        row_a = a[i]
        row_r = result[i]
        for p in range(n):
            value = row_a[p]
            if value == 0:
                continue
            row_b = b[p]
            for j in range(n):
                row_r[j] += value * row_b[j]
        if mod is not None:
            for j in range(n):
                row_r[j] %= mod
    return result


def matpow(matrix, exponent, mod=None, counter=None):
    """Binary exponentiation; ``counter`` records the number of matrix multiplications."""
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in matrix]
    while exponent > 0:
        if exponent & 1:
            result = matmul(result, base, mod)
            if counter is not None:
                counter[0] += 1
        exponent >>= 1
        if exponent:
            base = matmul(base, base, mod)
            if counter is not None:
                counter[0] += 1
    return result


def matpow_naive(matrix, exponent, mod=None):
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for _ in range(exponent):
        result = matmul(result, matrix, mod)
    return result


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matrix_apply(matrix, vector, mod=None):
    result = []
    for row in matrix:
        total = sum(value * component for (value, component) in zip(row, vector))
        result.append(total % mod if mod is not None else total)
    return result


def brute_force_walks(n, edges, k, source, simple_paths=False):
    """DFS enumeration of walks (or of simple paths) with exactly k edges from ``source``."""
    outgoing = [[] for _ in range(n)]
    for (u, v) in edges:
        outgoing[u].append(v)
    for neighbours in outgoing:
        neighbours.sort()
    count = [0]

    def walk(vertex, remaining, visited):
        if remaining == 0:
            count[0] += 1
            return
        for nxt in outgoing[vertex]:
            if simple_paths and (visited >> nxt) & 1:
                continue
            walk(nxt, remaining - 1, visited | (1 << nxt))

    walk(source, k, 1 << source)
    return count[0]


def all_paths_count(n, edges):
    """The number of non-empty simple vertex sequences linked by edges (a DAG enumeration)."""
    outgoing = [[] for _ in range(n)]
    for (u, v) in edges:
        outgoing[u].append(v)
    total = 0

    def walk(vertex):
        nonlocal total
        for nxt in outgoing[vertex]:
            total += 1
            walk(nxt)

    for start in range(n):
        walk(start)
    return total


def binomial(n, k):
    if k < 0 or k > n:
        return 0
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def directed_cycle(n):
    return n, [(i, (i + 1) % n) for i in range(n)]


def undirected_triangle():
    return 3, [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]


def main():
    # --- Three methods agree on random DAGs -----------------------------------------------
    for (n, density, seed) in ((5, 40, 1), (7, 55, 2), (8, 30, 3), (9, 70, 4), (6, 100, 5)):
        edges = random_dag(n, density, seed)[1]
        order = topological_order(n, edges)
        assert order is not None and sorted(order) == list(range(n)), "topological sort is valid"
        position = {vertex: index for (index, vertex) in enumerate(order)}
        for (u, v) in edges:
            assert position[u] < position[v], "every edge points forward in the order"
        matrix = adjacency_matrix(n, edges)
        for k in range(0, n + 2):
            power = matpow(matrix, k)
            column_sums = [sum(power[i][j] for i in range(n)) for j in range(n)]
            assert matrix_apply(transpose(power), [1] * n) == column_sums, \
                "the all-ones vector against the transposed power gives the column sums"
            vector, total = path_vector_dp(n, edges, k)
            assert vector == column_sums, (n, k)
            typed, total_typed = path_vector_dp_topological(n, edges, k)
            assert typed == vector and total_typed == total
            reversed_power = matpow(adjacency_matrix(n, [(v, u) for (u, v) in edges]), k)
            for source in range(n):
                # (A^k)[i][j] counts walks i -> j, so the row of the source lists paths by end,
                # and reversing every edge turns that row into the column of the same matrix.
                source_vector, source_total = path_vector_dp(n, edges, k, source)
                assert list(power[source]) == source_vector
                assert list(reversed_power[source]) == [row[source] for row in power]
                assert source_total == brute_force_walks(n, edges, k, source), \
                    "the DP matches brute-force enumeration on a DAG"
            if k >= n:
                assert total == 0 and all(col == 0 for col in column_sums), \
                    "a DAG has no path with n or more edges"

    # --- Paths of every length against the complete DAG's closed forms --------------------
    for n in (1, 2, 5, 8, 10):
        assert topological_order(n, complete_dag(n)[1]) is not None
        matrix = adjacency_matrix(n, complete_dag(n)[1])
        grand_total = 0
        for k in range(0, n):
            power = matpow(matrix, k)
            total = sum(sum(row) for row in power)
            assert total == binomial(n, k + 1), ("C(n, k+1) paths with k edges in total", n, k)
            for source in range(n):
                assert sum(power[source]) == binomial(n - 1 - source, k), "paths starting here"
                assert sum(row[source] for row in power) == binomial(source, k), \
                    "paths ending here: pick the k predecessors among the lower indices"
            grand_total += total
        # Summing over k = 0..n-1 counts the n trivial one-vertex paths as well, so the total is
        # the number of non-empty vertex subsets, 2^n - 1; all_paths_count only follows edges.
        assert grand_total == 2 ** n - 1, "every non-empty vertex subset is a path exactly once"
        assert grand_total == all_paths_count(*complete_dag(n)) + n
        assert matpow(matrix, n) == [[0] * n for _ in range(n)]

    # --- Degenerate lengths --------------------------------------------------------------
    n, edges = random_dag(6, 50, 11)
    matrix = adjacency_matrix(n, edges)
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    assert matpow(matrix, 0) == identity
    assert path_vector_dp(n, edges, 0) == ([1] * n, n), "k = 0 counts the n trivial paths"
    assert path_vector_dp(n, edges, 0, 2) == ([1 if v == 2 else 0 for v in range(n)], 1)
    assert all_paths_count(n, edges) + n == \
        sum(path_vector_dp(n, edges, k)[1] for k in range(n)), "the n trivial paths are included"
    assert path_vector_dp(1, [], 3) == ([0], 0)
    assert path_vector_dp(1, [], 0) == ([1], 1)

    # --- Fast exponentiation: exactness, modularity, multiplication count -----------------
    n, edges = random_dag(9, 60, 21)
    matrix = adjacency_matrix(n, edges)
    for k in (0, 1, 2, 3, 7, 11, 12, 13, 40):
        assert matpow(matrix, k) == matpow_naive(matrix, k), k
        fast_mod = matpow(matrix, k, MOD)
        assert fast_mod == [[value % MOD for value in row] for row in matpow(matrix, k)]
        _, total = path_vector_dp(n, edges, k)
        assert total % MOD == sum(sum(row) for row in fast_mod) % MOD
    counter = [0]
    matpow(matrix, 1000, None, counter)
    assert counter[0] <= 2 * 10, "1000 = 2^10 * 0.9766 takes at most 2 log2(1000) multiplications"
    assert counter[0] <= 2 * (1000).bit_length()
    assert matpow(matrix, 5) == matmul(matpow(matrix, 2), matpow(matrix, 3)), "A^(p+q) = A^p A^q"

    # --- The documented counterexample: walks are not paths off a DAG --------------------
    n, edges = directed_cycle(3)
    matrix = adjacency_matrix(n, edges)
    power = matpow(matrix, 3)
    assert sum(power[i][i] for i in range(n)) == 3, "3 closed walks of length 3 around the cycle"
    for source in range(n):
        assert brute_force_walks(n, edges, 3, source) == 1
        assert brute_force_walks(n, edges, 3, source, simple_paths=True) == 0, \
            "but no simple path of length 3 exists on 3 vertices"
    n, edges = undirected_triangle()
    matrix = adjacency_matrix(n, edges)
    assert sum(matpow(matrix, 3)[i][i] for i in range(n)) == 6, "6 closed walks of length 3"
    assert brute_force_walks(n, edges, 3, 0, simple_paths=True) == 0
    assert brute_force_walks(n, edges, 2, 0, simple_paths=True) == 2
    # On the DAG side the two notions coincide, which is the whole reason A^k can be used.
    n, edges = complete_dag(5)
    for k in range(0, 5):
        for source in range(n):
            assert brute_force_walks(n, edges, k, source) == \
                brute_force_walks(n, edges, k, source, simple_paths=True)

    # --- Determinism ---------------------------------------------------------------------
    n, edges = random_dag(9, 50, 31)
    for k in range(0, 6):
        first = (path_vector_dp(n, edges, k), path_vector_dp_topological(n, edges, k),
                 matpow(adjacency_matrix(n, edges), k))
        second = (path_vector_dp(n, edges, k), path_vector_dp_topological(n, edges, k),
                  matpow(adjacency_matrix(n, edges), k))
        assert first == second
    assert topological_order(n, edges) == topological_order(n, edges)
    assert path_vector_dp(n, edges, 2, 3) == path_vector_dp(n, edges, 2, 3)


if __name__ == "__main__":
    main()
    print("PASS")
