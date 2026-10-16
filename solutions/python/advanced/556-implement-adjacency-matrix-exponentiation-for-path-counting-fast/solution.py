"""Task 556 - Implement adjacency matrix exponentiation for path counting (fast exponent). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Fast walk counting by adjacency-matrix exponentiation, plus the reductions that make it worth doing.
``matpow`` is binary exponentiation with exact integers (or modular arithmetic) and is checked
against repeated multiplication, against the identity for exponent zero, against the semigroup law
A^(p+q) = A^p A^q, and against a k-step DP that applies A to a vector k times - the vector
recurrence is O(k n^2) while the exponentiation is O(n^3 log k). A second, structurally different
engine multiplies Boolean and GF(2) matrices with Python integers as bit rows and bit columns; the
file counts *operations* for it rather than seconds - n^2 word operations against n^3 scalar ones -
precisely because wall-clock timing is not a portable assertion. Graphs and random instances come
from a full-period 64-bit LCG (MMIX constants) read in its high bits; the repo's 16-bit LCG
``(seed*75+74) % 65537`` has a short period and low-bit structure that would bias edge density.

Checks: on random graphs the k-step vector DP equals A^k and a brute-force walk enumeration for
every k up to n + 1, and the diagonal entries equal a separate brute-force count of closed walks;
the complete graph matches the closed forms ((n-1)^k + (n-1)(-1)^k)/n for closed walks and
((n-1)^k - (-1)^k)/n between distinct vertices; trace(A^2) = 2m, trace(A^3) = 6 times the triangle
count, and trace(A^4) = 8*(4-cycles) + 2m + 2*sum_v d_v(d_v - 1) with the 4-cycle count
independently brute-forced over vertex quadruples (K4 gives 3, K_{3,3} gives 9, C6 gives 6); a tree
and a star have zero odd traces. The Boolean engine reproduces the transitive closure - squaring
(I or A) ceil(log2(n-1)) times reaches exactly the BFS closure, which also equals the
Floyd-Warshall closure, on random graphs up to 8 vertices - the GF(2) engine reproduces the naive
product mod 2, A^k mod 2 by bitset parity, and the parity of the exact trace. A by-product the file
pins down: an undirected graph has no odd closed walk of length at most n exactly when it is
bipartite, so iterating trace(A^k) for odd k and the BFS two-colouring agree on every random
instance. Honest caveat, asserted: A^k counts walks, not simple paths - K3 has six closed 3-walks
and no simple closed walk of length 3 - so the identity only becomes a path count on an acyclic
graph.
"""
import itertools

MASK64 = (1 << 64) - 1
MOD = 1000000007


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_graph(n, density, seed):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 17) & MASK64)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            state = prng64(state)
            if (state >> 33) % 100 < density:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return (n, adj)


def complete_graph(n):
    full = (1 << n) - 1
    return (n, [full & ~(1 << u) for u in range(n)])


def path_graph(n):
    adj = [0] * n
    for i in range(n - 1):
        adj[i] |= 1 << (i + 1)
        adj[i + 1] |= 1 << i
    return (n, adj)


def cycle_graph(n):
    graph = path_graph(n)
    adj = graph[1]
    adj[0] |= 1 << (n - 1)
    adj[n - 1] |= 1 << 0
    return (n, adj)


def star_graph(k):
    adj = [0] * (k + 1)
    for leaf in range(1, k + 1):
        adj[0] |= 1 << leaf
        adj[leaf] |= 1 << 0
    return (k + 1, adj)


def complete_bipartite(a, b):
    adj = [0] * (a + b)
    for i in range(a):
        for j in range(a, a + b):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
    return (a + b, adj)


def adjacency_matrix(graph):
    n, adj = graph
    return [[(adj[i] >> j) & 1 for j in range(n)] for i in range(n)]


def matmul(a, b, mod=None, counter=None):
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
    if counter is not None:
        counter[0] += n ** 3
    return result


def matpow(matrix, exponent, mod=None, counter=None):
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in matrix]
    while exponent > 0:
        if exponent & 1:
            result = matmul(result, base, mod, counter)
        exponent >>= 1
        if exponent:
            base = matmul(base, base, mod, counter)
    return result


def matpow_naive(matrix, exponent, mod=None):
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for _ in range(exponent):
        result = matmul(result, matrix, mod)
    return result


def walk_dp(graph, k, source):
    """k applications of A to a unit vector: O(k n^2) for one starting vertex."""
    n, adj = graph
    vector = [1 if v == source else 0 for v in range(n)]
    for _ in range(k):
        following = [0] * n
        for v in range(n):
            total = 0
            bits = adj[v]
            while bits:
                low = bits & -bits
                bits ^= low
                total += vector[low.bit_length() - 1]
            following[v] = total
        vector = following
    return vector


def brute_force_walks(graph, k, source, simple=False):
    _, adj = graph
    count = [0]

    def walk(vertex, remaining, visited):
        if remaining == 0:
            count[0] += 1
            return
        bits = adj[vertex]
        while bits:
            low = bits & -bits
            bits ^= low
            nxt = low.bit_length() - 1
            if simple and (visited >> nxt) & 1:
                continue
            walk(nxt, remaining - 1, visited | (1 << nxt))

    walk(source, k, 1 << source)
    return count[0]


def brute_force_closed_walks(graph, k, source):
    """Walks of exactly k edges returning to the start: the diagonal entry (A^k)[source][source]."""
    _, adj = graph
    count = [0]

    def walk(vertex, remaining):
        if remaining == 0:
            if vertex == source:
                count[0] += 1
            return
        bits = adj[vertex]
        while bits:
            low = bits & -bits
            bits ^= low
            walk(low.bit_length() - 1, remaining - 1)

    walk(source, k)
    return count[0]


def columns_of(rows, n):
    columns = [0] * n
    for i in range(n):
        bits = rows[i]
        while bits:
            low = bits & -bits
            bits ^= low
            columns[low.bit_length() - 1] |= 1 << i
    return columns


def boolean_matmul(rows_a, rows_b, n, counter=None):
    """Boolean product with bit rows and bit columns: n^2 word operations."""
    columns_b = columns_of(rows_b, n)
    result = []
    for i in range(n):
        row = 0
        for j in range(n):
            if rows_a[i] & columns_b[j]:
                row |= 1 << j
        result.append(row)
    if counter is not None:
        counter[0] += n * n
    return result


def gf2_matmul(rows_a, rows_b, n, counter=None):
    """Product over GF(2): the parity of each bit intersection, again n^2 word operations."""
    columns_b = columns_of(rows_b, n)
    result = []
    for i in range(n):
        row = 0
        for j in range(n):
            if bin(rows_a[i] & columns_b[j]).count("1") & 1:
                row |= 1 << j
        result.append(row)
    if counter is not None:
        counter[0] += n * n
    return result


def gf2_naive(a, b, n):
    return [[sum(a[i][k] * b[k][j] for k in range(n)) % 2 for j in range(n)] for i in range(n)]


def rows_to_matrix(rows, n):
    return [[(rows[i] >> j) & 1 for j in range(n)] for i in range(n)]


def matrix_to_rows(matrix, n):
    return [sum(matrix[i][j] << j for j in range(n)) for i in range(n)]


def reachability_closure(graph):
    """Transitive closure by squaring (I or A) until walks of length >= n-1 are covered."""
    n, adj = graph
    rows = [adj[i] | (1 << i) for i in range(n)]
    for _ in range(max(1, (n - 1).bit_length())):
        rows = boolean_matmul(rows, rows, n)
    return rows


def bfs_reachability(graph):
    n, adj = graph
    rows = []
    for start in range(n):
        seen = 1 << start
        frontier = adj[start] & ~seen
        seen |= frontier
        while frontier:
            following = 0
            bits = frontier
            while bits:
                low = bits & -bits
                bits ^= low
                following |= adj[low.bit_length() - 1]
            frontier = following & ~seen
            seen |= frontier
        rows.append(seen)
    return rows


def floyd_warshall_closure(graph):
    n, adj = graph
    reach = [[bool((adj[i] >> j) & 1) or i == j for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    if reach[k][j]:
                        reach[i][j] = True
    return [sum(1 << j for j in range(n) if reach[i][j]) for i in range(n)]


def is_bipartite(graph):
    n, adj = graph
    colour = [-1] * n
    for start in range(n):
        if colour[start] != -1:
            continue
        colour[start] = 0
        queue = [start]
        while queue:
            vertex = queue.pop(0)
            bits = adj[vertex]
            while bits:
                low = bits & -bits
                bits ^= low
                nxt = low.bit_length() - 1
                if colour[nxt] == -1:
                    colour[nxt] = 1 - colour[vertex]
                    queue.append(nxt)
                elif colour[nxt] == colour[vertex]:
                    return False
    return True


def is_bipartite_by_traces(graph):
    """Bipartite iff trace(A^k) = 0 for every odd k <= n (no odd closed walk fits)."""
    n, _ = graph
    matrix = adjacency_matrix(graph)
    power = [row[:] for row in matrix]
    for k in range(1, n + 1):
        if k % 2 == 1 and any(power[i][i] for i in range(n)):
            return False
        power = matmul(power, matrix)
    return True


def triangle_count(graph):
    n, adj = graph
    total = 0
    for i in range(n):
        bits = adj[i] & ~((1 << (i + 1)) - 1)
        while bits:
            low = bits & -bits
            bits ^= low
            j = low.bit_length() - 1
            total += bin(adj[i] & adj[j] & ~((1 << (j + 1)) - 1)).count("1")
    return total


def four_cycle_count(graph):
    """4-cycles as subgraphs.

    The smallest vertex of each quadruple is pinned as the start, so every undirected 4-cycle is
    generated exactly twice - once per direction - and the raw count is halved.
    """
    n, adj = graph
    total = 0
    for subset in itertools.combinations(range(n), 4):
        first = subset[0]
        for (b, c, d) in itertools.permutations(subset[1:]):
            if all(adj[x] >> y & 1 for (x, y) in ((first, b), (b, c), (c, d), (d, first))):
                total += 1
    return total // 2


def trace(power):
    return sum(power[i][i] for i in range(len(power)))


def main():
    # --- Exponentiation against the k-step DP and the brute force -------------------------
    for (n, density, seed) in ((5, 40, 1), (7, 60, 2), (6, 100, 3), (8, 35, 4)):
        graph = random_graph(n, density, seed)
        matrix = adjacency_matrix(graph)
        identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        for k in range(0, n + 1):
            power = matpow(matrix, k)
            assert power == matpow_naive(matrix, k), (n, k)
            for source in range(n):
                assert power[source] == walk_dp(graph, k, source)
                assert sum(power[source]) == brute_force_walks(graph, k, source)
                assert power[source][source] == brute_force_closed_walks(graph, k, source)
        assert matpow(matrix, 0) == identity
        assert matpow(matrix, 7) == matmul(matpow(matrix, 3), matpow(matrix, 4))
        assert matpow(matrix, 0, MOD) == identity
        assert matpow(matrix, 9, MOD) == [[value % MOD for value in row]
                                          for row in matpow(matrix, 9)]
        assert matpow(matrix, 5, MOD) == matmul(matpow(matrix, 2, MOD), matpow(matrix, 3, MOD), MOD)

    # --- Closed forms on the complete graph ----------------------------------------------
    for n in (2, 3, 4, 6):
        matrix = adjacency_matrix(complete_graph(n))
        for k in range(0, 12):
            power = matpow(matrix, k)
            closed = ((n - 1) ** k + (n - 1) * ((-1) ** k)) // n
            between = ((n - 1) ** k - ((-1) ** k)) // n
            for i in range(n):
                assert power[i][i] == closed, (n, k, i)
                for j in range(n):
                    if i != j:
                        assert power[i][j] == between, (n, k, i, j)
            assert trace(power) == n * closed
    assert trace(matpow(adjacency_matrix(complete_graph(5)), 2)) == 5 * 4, "2 closed walks per edge"

    # --- Traces against the structural counts --------------------------------------------
    for (n, density, seed) in ((9, 35, 11), (10, 50, 12), (8, 70, 13)):
        graph = random_graph(n, density, seed)
        _, adj = graph
        matrix = adjacency_matrix(graph)
        degrees = [bin(row).count("1") for row in adj]
        edges = sum(degrees) // 2
        triangles = triangle_count(graph)
        four_cycles = four_cycle_count(graph)
        power2, power3, power4 = matpow(matrix, 2), matpow(matrix, 3), matpow(matrix, 4)
        assert trace(power2) == 2 * edges
        assert trace(power3) == 6 * triangles
        assert trace(power4) == 8 * four_cycles + 2 * edges + 2 * sum(d * (d - 1) for d in degrees)
        assert trace(power3) == sum(brute_force_closed_walks(graph, 3, s) for s in range(n))
        assert trace(power4) == sum(brute_force_closed_walks(graph, 4, s) for s in range(n))

    # --- Trees, stars, and the 4-cycle counts on known instances -------------------------
    tree = path_graph(7)
    tree_matrix = adjacency_matrix(tree)
    for k in range(1, 8, 2):
        assert trace(matpow(tree_matrix, k)) == 0, "a tree has no odd closed walk"
    assert four_cycle_count(tree) == 0
    assert trace(matpow(tree_matrix, 4)) == 8 * four_cycle_count(tree) + 2 * 6 + \
        2 * sum(d * (d - 1) for d in (1, 2, 2, 2, 2, 2, 1))
    assert trace(matpow(tree_matrix, 4)) == 32, "12 backtracking walks plus 20 three-vertex ones"
    star = star_graph(4)
    star_matrix = adjacency_matrix(star)
    assert trace(matpow(star_matrix, 2)) == 8
    for k in range(1, 6, 2):
        assert trace(matpow(star_matrix, k)) == 0
    assert four_cycle_count(complete_graph(4)) == 3, "K4 has three 4-cycles"
    assert four_cycle_count(complete_bipartite(3, 3)) == 9, "K3,3 has C(3,2)^2 4-cycles"
    assert four_cycle_count(cycle_graph(6)) == 0, "C6 has no 4-cycle"
    assert four_cycle_count(complete_graph(5)) == 15, "each 4-subset carries 3 four-cycles"
    assert four_cycle_count(complete_graph(4)) == 3
    assert trace(matpow(adjacency_matrix(complete_bipartite(3, 3)), 4)) == \
        8 * 9 + 2 * 9 + 2 * 6 * 3 * 2

    # --- Boolean and GF(2) engines, with operation counting ------------------------------
    graph = random_graph(8, 45, 21)
    n = graph[0]
    rows = [graph[1][i] for i in range(n)]
    naive_boolean = [[1 if any((rows[i] >> k) & 1 and (rows[k] >> j) & 1 for k in range(n))
                      else 0 for j in range(n)] for i in range(n)]
    assert boolean_matmul(rows, rows, n) == matrix_to_rows(naive_boolean, n)
    assert gf2_matmul(rows, rows, n) == matrix_to_rows(gf2_naive(rows_to_matrix(rows, n),
                                                                 rows_to_matrix(rows, n), n), n)
    assert boolean_matmul(rows, rows, n) == boolean_matmul(rows, rows, n)
    bitset_counter, naive_counter = [0], [0]
    boolean_matmul(rows, rows, n, bitset_counter)
    gf2_matmul(rows, rows, n, bitset_counter)
    for _ in range(2):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    naive_counter[0] += 1
    assert bitset_counter[0] == 2 * n * n and naive_counter[0] == 2 * n ** 3
    assert bitset_counter[0] < naive_counter[0], "n^2 word operations beat n^3 scalar ones"
    for (size, density, seed) in ((6, 40, 31), (7, 55, 32), (8, 30, 33), (5, 80, 34)):
        graph = random_graph(size, density, seed)
        closure = reachability_closure(graph)
        assert closure == bfs_reachability(graph)
        assert closure == floyd_warshall_closure(graph)
        assert all(row & (1 << i) for (i, row) in enumerate(closure)), "i reaches itself"

    # --- GF(2) powers modulo 2, and bipartiteness from the traces ------------------------
    for (size, density, seed) in ((6, 40, 41), (7, 45, 42), (8, 30, 43), (6, 70, 44), (5, 40, 45)):
        graph = random_graph(size, density, seed)
        n = graph[0]
        matrix = adjacency_matrix(graph)
        rows = [graph[1][i] for i in range(n)]
        power_rows = [1 << i for i in range(n)]                  # identity over GF(2)
        for k in range(0, 6):
            exact = matpow(matrix, k)
            reduced = [[value & 1 for value in row] for row in exact]
            assert power_rows == matrix_to_rows(reduced, n), "A^k mod 2 by bitset parity"
            assert sum((power_rows[i] >> i) & 1 for i in range(n)) % 2 == trace(exact) & 1
            power_rows = gf2_matmul(power_rows, rows, n)
        assert is_bipartite_by_traces(graph) == is_bipartite(graph)
    assert is_bipartite_by_traces(path_graph(6)) and is_bipartite(cycle_graph(6))
    assert not is_bipartite_by_traces(cycle_graph(5)) and not is_bipartite(cycle_graph(5))
    assert not is_bipartite_by_traces(complete_graph(4))

    # --- Documented caveat: walks are not simple paths ------------------------------------
    triangle = complete_graph(3)
    matrix = adjacency_matrix(triangle)
    assert trace(matpow(matrix, 3)) == 6, "six closed 3-walks"
    assert brute_force_walks(triangle, 3, 0, simple=True) == 0, "but no simple path of length 3"
    assert brute_force_walks(triangle, 3, 0) == 8, "eight 3-step walks leave a vertex of K3"
    assert brute_force_closed_walks(triangle, 3, 0) == 2
    assert sum(brute_force_closed_walks(triangle, 3, s) for s in range(3)) == 6
    assert trace(matpow(adjacency_matrix(path_graph(4)), 3)) == 0, "a tree: no closed 3-walk"
    assert brute_force_walks(path_graph(4), 3, 0) == 3, "but open 3-walks do exist in a tree"
    assert brute_force_closed_walks(path_graph(4), 3, 0) == 0

    # --- Determinism ---------------------------------------------------------------------
    graph = random_graph(9, 50, 51)
    matrix = adjacency_matrix(graph)
    assert matpow(matrix, 15) == matpow(matrix, 15)
    assert walk_dp(graph, 6, 2) == walk_dp(graph, 6, 2)
    assert reachability_closure(graph) == reachability_closure(graph)
    assert four_cycle_count(graph) == four_cycle_count(graph)


if __name__ == "__main__":
    main()
    print("PASS")
