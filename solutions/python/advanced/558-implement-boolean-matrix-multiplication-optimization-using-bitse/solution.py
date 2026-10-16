"""Task 558 - Implement boolean matrix multiplication optimization using bitsets. (Tier: Advanced, CTCI Ch 5 — Bit Manipulation)

Boolean and GF(2) matrix multiplication with Python integers as bit rows and bit columns. The
optimized product walks the set bits of each row of A and ORs the corresponding columns of B (over
GF(2) it XORs them) instead of looping over every (i, k, j) triple, turning n^3 scalar bit
operations into at most n^2 word-sized ones - and with n bits per word that is n^3/64 bit
operations. The file counts the actual OR/XOR word operations and compares them with n^3 instead of
clocking anything, because wall-clock timing is not a portable assertion. Three applications follow
from the same engine: Boolean powers, where (I ∨ A)^k has a 1 exactly when there is a walk of at
most k edges, so the smallest such k is the BFS distance and ceil(log2) squarings give the
transitive closure; Warshall's algorithm with bit rows, one row OR per set bit instead of n^3
scalar steps; and GF(2) powers, where the entries of A^k give walk counts modulo 2. Random
matrices come from a full-period 64-bit LCG (MMIX constants) read in its high bits, since the
repo's 16-bit LCG ``(seed*75+74) % 65537`` has a short period and low-bit structure that would bias
the density.

Checks: the bitset product equals the naive product on random matrices of several sizes and
densities and on structured ones (identity as a two-sided unit, all-ones, upper triangular, zero,
and permutation matrices, where the product must equal the composition of the permutations); the
GF(2) engine equals the naive product modulo 2; the word-operation count equals the number of set
bits consumed and stays below n^3, and the same holds for Warshall against its n^3 version; for
every k the k-th Boolean power of I ∨ A equals an independent k-step reachability DP, the smallest
k with a 1 entry equals the BFS distance, and squaring reaches exactly the closure found by bitset
Warshall, naive Warshall and BFS; and on random *directed* graphs the first power with a set
diagonal entry agrees with DFS colouring. Honest caveats, all three asserted: Boolean and GF(2) products
genuinely differ - on K4 the Boolean square is all ones while the GF(2) square is the
identity, because the two 2-walks between distinct vertices cancel mod 2 and the diagonal is the
odd degree; GF(2) arithmetic cannot count
triangles at all, because the diagonal of A^3 vanishes identically (every closed 3-walk cancels
with its reversal) while the integer trace is 6 times the triangle count; and the diagonal
criterion does not detect cycles in an undirected graph, where every edge is already a closed
2-walk, so a tree passes it while containing no undirected cycle.
"""
MASK64 = (1 << 64) - 1


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_matrix(n, density, seed):
    """Boolean matrix with rows as bit masks."""
    state = prng64((seed * 0x9E3779B97F4A7C15 + 23) & MASK64)
    rows = []
    for i in range(n):
        row = 0
        for j in range(n):
            state = prng64(state)
            if (state >> 33) % 100 < density:
                row |= 1 << j
        rows.append(row)
    return rows


def random_graph(n, density, seed):
    """Undirected graph with rows as bit masks."""
    state = prng64((seed * 0x9E3779B97F4A7C15 + 29) & MASK64)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            state = prng64(state)
            if (state >> 33) % 100 < density:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return (n, adj)


def random_digraph(n, density, seed):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 31) & MASK64)
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            state = prng64(state)
            if (state >> 33) % 100 < density:
                adj[i] |= 1 << j
    return (n, adj)


def columns_of(rows, n):
    columns = [0] * n
    for i in range(n):
        bits = rows[i]
        while bits:
            low = bits & -bits
            bits ^= low
            columns[low.bit_length() - 1] |= 1 << i
    return columns


def bitset_boolean_matmul(rows_a, rows_b, n, counter=None):
    """Row i of the product is the OR of the rows of B selected by the set bits of row i of A."""
    result = []
    for i in range(n):
        row = 0
        bits = rows_a[i]
        while bits:
            low = bits & -bits
            bits ^= low
            row |= rows_b[low.bit_length() - 1]
            if counter is not None:
                counter[0] += 1
        result.append(row)
    return result


def bitset_boolean_matmul_by_k(rows_a, rows_b, n):
    """The same product with the loops exchanged: each row of B is folded in once."""
    result = [0] * n
    for k in range(n):
        row_b = rows_b[k]
        bit = 1 << k
        for i in range(n):
            if rows_a[i] & bit:
                result[i] |= row_b
    return result


def bitset_boolean_matmul_by_columns(rows_a, rows_b, n):
    """The same product again, via (A B)^T = B^T A^T using the columns of both inputs."""
    return columns_of(bitset_boolean_matmul(columns_of(rows_b, n), columns_of(rows_a, n), n), n)


def bitset_gf2_matmul(rows_a, rows_b, n, counter=None):
    """XOR of the selected rows of B: the same product over GF(2)."""
    result = []
    for i in range(n):
        row = 0
        bits = rows_a[i]
        while bits:
            low = bits & -bits
            bits ^= low
            row ^= rows_b[low.bit_length() - 1]
            if counter is not None:
                counter[0] += 1
        result.append(row)
    return result


def naive_boolean_matmul(rows_a, rows_b, n, counter=None):
    result = []
    for i in range(n):
        row = 0
        for j in range(n):
            witness = False
            for k in range(n):
                if counter is not None:
                    counter[0] += 1
                if (rows_a[i] >> k) & 1 and (rows_b[k] >> j) & 1:
                    witness = True
            if witness:
                row |= 1 << j
        result.append(row)
    return result


def naive_gf2_matmul(rows_a, rows_b, n, counter=None):
    result = []
    for i in range(n):
        row = 0
        for j in range(n):
            parity = 0
            for k in range(n):
                if counter is not None:
                    counter[0] += 1
                parity ^= ((rows_a[i] >> k) & 1) & ((rows_b[k] >> j) & 1)
            if parity:
                row |= 1 << j
        result.append(row)
    return result


def identity_rows(n):
    return [1 << i for i in range(n)]


def walk_counts(rows, n, k):
    """Exact number of i -> j walks with k edges, by the k-step DP (integers)."""
    counts = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for _ in range(k):
        following = [[0] * n for _ in range(n)]
        for i in range(n):
            for middle in range(n):
                value = counts[i][middle]
                if not value:
                    continue
                for j in range(n):
                    if (rows[middle] >> j) & 1:
                        following[i][j] += value
        counts = following
    return counts


def permutation_rows(perm):
    return [1 << image for image in perm]


def bitset_warshall(rows, n, counter=None):
    """Warshall with bit rows: one row OR per set bit of the pivot mask."""
    reach = [row | (1 << i) for (i, row) in enumerate(rows)]
    for k in range(n):
        bit = 1 << k
        row_k = reach[k]
        for i in range(n):
            if reach[i] & bit:
                if counter is not None:
                    counter[0] += 1
                reach[i] |= row_k
    return reach


def naive_warshall(rows, n, counter=None):
    reach = [[bool((rows[i] >> j) & 1) or i == j for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if counter is not None:
                    counter[0] += 1
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
    return [sum(reach[i][j] << j for j in range(n)) for i in range(n)]


def within_k_steps_dp(graph, k):
    """(I ∨ A)^k computed by the plain k-step reachability recurrence, without any product."""
    n, adj = graph
    step = [adj[i] | (1 << i) for i in range(n)]
    reach = identity_rows(n)
    for _ in range(k):
        following = list(reach)
        for i in range(n):
            bits = reach[i]
            while bits:
                low = bits & -bits
                bits ^= low
                following[i] |= step[low.bit_length() - 1]
        reach = following
    return reach


def bfs_distances(graph):
    """Shortest path lengths as integers, INF when unreachable."""
    n, adj = graph
    distances = []
    for start in range(n):
        distance = [None] * n
        distance[start] = 0
        frontier = [start]
        while frontier:
            following = []
            for vertex in frontier:
                bits = adj[vertex]
                while bits:
                    low = bits & -bits
                    bits ^= low
                    nxt = low.bit_length() - 1
                    if distance[nxt] is None:
                        distance[nxt] = distance[vertex] + 1
                        following.append(nxt)
            frontier = following
        distances.append(distance)
    return distances


def closure_by_squaring(rows, n):
    """(I ∨ A)^(2^t) with t = ceil(log2(n-1)) is the transitive closure."""
    current = [row | (1 << i) for (i, row) in enumerate(rows)]
    for _ in range(max(1, (n - 1).bit_length())):
        current = bitset_boolean_matmul(current, current, n)
    return current


def has_cycle_dfs(graph):
    """Cycle detection in a directed graph by three-colouring (recursive)."""
    n, adj = graph
    colour = [0] * n

    def visit(vertex):
        colour[vertex] = 1
        bits = adj[vertex]
        while bits:
            low = bits & -bits
            bits ^= low
            nxt = low.bit_length() - 1
            if colour[nxt] == 1:
                return True
            if colour[nxt] == 0 and visit(nxt):
                return True
        colour[vertex] = 2
        return False

    for v in range(n):
        if colour[v] == 0 and visit(v):
            return True
    return False


def cycle_by_boolean_powers(graph):
    """A directed cycle exists iff some A^k with 1 <= k <= n has a set diagonal entry."""
    n, adj = graph
    power = list(adj)
    for _ in range(1, n + 1):
        if any((power[i] >> i) & 1 for i in range(n)):
            return True
        power = bitset_boolean_matmul(power, adj, n)
    return False


def undirected_cycle(graph):
    """A graph is a forest iff edges = vertices - components; a cycle means strictly more."""
    n, adj = graph
    edges = sum(bin(row).count("1") for row in adj) // 2
    seen = 0
    components = 0
    for start in range(n):
        if (seen >> start) & 1:
            continue
        components += 1
        stack = [start]
        seen |= 1 << start
        while stack:
            vertex = stack.pop()
            bits = adj[vertex] & ~seen
            while bits:
                low = bits & -bits
                bits ^= low
                seen |= low
                stack.append(low.bit_length() - 1)
    return edges > n - components


def trace_exact(rows, n, k):
    """trace(A^k) over the integers, from the exact walk counts."""
    counts = walk_counts(rows, n, k)
    return sum(counts[i][i] for i in range(n))


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


def path_graph(n):
    adj = [0] * n
    for i in range(n - 1):
        adj[i] |= 1 << (i + 1)
        adj[i + 1] |= 1 << i
    return (n, adj)


def complete_graph(n):
    full = (1 << n) - 1
    return (n, [full & ~(1 << u) for u in range(n)])


def cycle_graph(n):
    graph = path_graph(n)
    adj = graph[1]
    adj[0] |= 1 << (n - 1)
    adj[n - 1] |= 1 << 0
    return (n, adj)


def main():
    # --- The bitset product equals the naive one ------------------------------------------
    for n in (1, 2, 3, 5, 8):
        for seed in (1, 2, 3):
            a = random_matrix(n, 45, seed)
            b = random_matrix(n, 45, seed + 100)
            assert bitset_boolean_matmul(a, b, n) == naive_boolean_matmul(a, b, n), (n, seed)
            assert bitset_boolean_matmul_by_k(a, b, n) == naive_boolean_matmul(a, b, n)
            assert bitset_boolean_matmul_by_columns(a, b, n) == naive_boolean_matmul(a, b, n)
            assert bitset_gf2_matmul(a, b, n) == naive_gf2_matmul(a, b, n), (n, seed)
    n = 6
    samples = [random_matrix(n, density, 200 + density) for density in (0, 10, 50, 90, 100)]
    for a in samples:
        for b in samples:
            assert bitset_boolean_matmul(a, b, n) == naive_boolean_matmul(a, b, n)
            assert bitset_gf2_matmul(a, b, n) == naive_gf2_matmul(a, b, n)
    for (n, density, seed) in ((4, 40, 61), (6, 60, 62), (7, 30, 63)):
        a = random_matrix(n, density, seed)
        b = random_matrix(n, density, seed + 1)
        c = random_matrix(n, density, seed + 2)
        assert bitset_boolean_matmul(bitset_boolean_matmul(a, b, n), c, n) == \
            bitset_boolean_matmul(a, bitset_boolean_matmul(b, c, n), n), "associativity"

    # --- Structured matrices ---------------------------------------------------------------
    n = 5
    identity = identity_rows(n)
    ones = [(1 << n) - 1] * n
    zero = [0] * n
    upper = [sum(1 << j for j in range(i, n)) for i in range(n)]
    a = random_matrix(n, 60, 7)
    assert bitset_boolean_matmul(a, identity, n) == a == bitset_boolean_matmul(identity, a, n)
    assert bitset_gf2_matmul(a, identity, n) == a == bitset_gf2_matmul(identity, a, n)
    assert bitset_boolean_matmul(a, zero, n) == zero
    assert bitset_boolean_matmul(ones, ones, n) == ones, "a 2-walk exists for every pair"
    assert bitset_boolean_matmul(upper, upper, n) == upper, "i < j <= k chains stay upper"
    first = permutation_rows([1, 2, 0, 4, 3])
    second = permutation_rows([4, 0, 3, 2, 1])
    composed = bitset_boolean_matmul(first, second, n)
    expected = permutation_rows([second[first[i].bit_length() - 1].bit_length() - 1
                                 for i in range(n)])
    assert composed == expected, "permutation matrices compose as their permutations do"
    assert bitset_boolean_matmul(first, second, n) == \
        bitset_gf2_matmul(first, second, n), "a permutation matrix has a single witness per entry"

    # --- Word operations against scalar operations ----------------------------------------
    n = 8
    a = random_matrix(n, 50, 11)
    b = random_matrix(n, 50, 12)
    word_counter, scalar_counter = [0], [0]
    bitset_boolean_matmul(a, b, n, word_counter)
    naive_boolean_matmul(a, b, n, scalar_counter)
    assert word_counter[0] == sum(bin(row).count("1") for row in a)
    assert word_counter[0] <= n * n < n ** 3 == scalar_counter[0]
    gf_counter = [0]
    bitset_gf2_matmul(a, b, n, gf_counter)
    assert gf_counter[0] == word_counter[0]
    warshall_counter, warshall_scalar = [0], [0]
    adjacency = random_graph(n, 40, 13)[1]
    assert bitset_warshall(adjacency, n, warshall_counter) == \
        naive_warshall(adjacency, n, warshall_scalar)
    assert warshall_counter[0] <= n * n < warshall_scalar[0] == n ** 3

    # --- Boolean powers against the k-step DP, the BFS distances and the closure ----------
    for (size, density, seed) in ((5, 45, 21), (6, 70, 22), (7, 30, 23)):
        graph = random_graph(size, density, seed)
        n, adj = graph
        step = [adj[i] | (1 << i) for i in range(n)]
        distances = bfs_distances(graph)
        power = identity_rows(n)
        for k in range(0, size + 2):
            assert power == within_k_steps_dp(graph, k), (size, k)
            for i in range(n):
                for j in range(n):
                    assert bool((power[i] >> j) & 1) == (distances[i][j] is not None
                                                        and distances[i][j] <= k), (size, k, i, j)
            power = bitset_boolean_matmul(power, step, n)
        closure = closure_by_squaring(adj, n)
        assert closure == bitset_warshall(adj, n) == naive_warshall(adj, n)
        for i in range(n):
            for j in range(n):
                reachable = distances[i][j] is not None
                assert bool((closure[i] >> j) & 1) == reachable

    # --- GF(2) powers, triangle parity, and directed cycles ------------------------------
    for (size, density, seed) in ((5, 50, 31), (6, 60, 32), (7, 40, 33)):
        graph = random_graph(size, density, seed)
        n, adj = graph
        bitset_power = identity_rows(n)
        naive_power = identity_rows(n)
        for k in range(0, 5):
            assert bitset_power == naive_power, (size, k)
            bitset_power = bitset_gf2_matmul(bitset_power, adj, n)
            naive_power = naive_gf2_matmul(naive_power, adj, n)
        third = bitset_gf2_matmul(bitset_gf2_matmul(adj, adj, n), adj, n)
        exact = walk_counts(adj, n, 3)
        for i in range(n):
            for j in range(n):
                assert (third[i] >> j) & 1 == exact[i][j] % 2, "A^3 over GF(2) is the parity"
        assert sum((third[i] >> i) & 1 for i in range(n)) == 0, \
            "the diagonal of A^3 over GF(2) vanishes: closed 3-walks cancel with their reversals"
        assert trace_exact(adj, n, 3) == 6 * triangle_count(graph), \
            "so the integer trace is what sees triangles"
        second = bitset_gf2_matmul(adj, adj, n)
        assert trace_exact(adj, n, 2) == sum(bin(row).count("1") for row in adj), \
            "the integer diagonal of A^2 sums to 2m"
        for i in range(n):
            assert (second[i] >> i) & 1 == bin(adj[i]).count("1") % 2, \
                "the GF(2) diagonal of A^2 is the parity of the degree"
    for (size, density, seed) in ((4, 40, 41), (5, 55, 42), (6, 35, 43), (5, 20, 44)):
        digraph = random_digraph(size, density, seed)
        assert cycle_by_boolean_powers(digraph) == has_cycle_dfs(digraph), (size, density, seed)
    assert has_cycle_dfs((3, [0b010, 0b100, 0b001])), "the directed 3-cycle"
    assert not has_cycle_dfs((3, [0b110, 0b100, 0b000])), "0 -> 1, 0 -> 2, 1 -> 2 is acyclic"
    assert cycle_by_boolean_powers((3, [0b010, 0b100, 0b001]))
    assert not cycle_by_boolean_powers((3, [0b110, 0b100, 0b000]))
    assert cycle_by_boolean_powers((2, [0b01, 0b01])), "a self-loop is a cycle"

    # --- Documented caveat 1: Boolean and GF(2) products differ ---------------------------
    n, adj = complete_graph(4)
    assert bitset_boolean_matmul(adj, adj, n) == [(1 << n) - 1] * n, \
        "every pair is joined by a 2-walk"
    assert bitset_gf2_matmul(adj, adj, n) == identity_rows(n), \
        "but mod 2 the two 2-walks between distinct vertices cancel and the diagonal is the " \
        "odd degree, so the GF(2) square is the identity"

    # --- Documented caveat 2: no cycle test for undirected graphs ------------------------
    tree = path_graph(4)
    n, adj = tree
    assert not undirected_cycle(tree)
    second = bitset_boolean_matmul(adj, adj, n)
    assert all((second[i] >> i) & 1 for i in range(n) if adj[i]), \
        "every edge is already a closed 2-walk, so the diagonal test fires on an acyclic graph"
    assert cycle_by_boolean_powers(tree), "which is why the criterion needs a directed graph"
    assert undirected_cycle(cycle_graph(4))
    assert cycle_by_boolean_powers(cycle_graph(4)) and undirected_cycle(complete_graph(4))

    # --- Determinism ---------------------------------------------------------------------
    a = random_matrix(7, 50, 51)
    assert bitset_boolean_matmul(a, a, 7) == bitset_boolean_matmul(a, a, 7)
    assert bitset_gf2_matmul(a, a, 7) == bitset_gf2_matmul(a, a, 7)
    graph = random_graph(7, 40, 52)
    assert bitset_warshall(graph[1], 7) == bitset_warshall(graph[1], 7)
    assert bfs_distances(graph) == bfs_distances(graph)


if __name__ == "__main__":
    main()
    print("PASS")
