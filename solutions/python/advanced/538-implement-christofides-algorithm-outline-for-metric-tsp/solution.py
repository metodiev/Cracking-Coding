"""Task 538 - Implement Christofides algorithm outline for metric TSP. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)

Christofides' 3/2-approximation for the metric travelling salesman, implemented in full for small
complete graphs. The four stages are: (1) a minimum spanning tree, computed twice with independent
algorithms - Kruskal over edges sorted by (weight, u, v) with a union-find, and Prim with a binary
heap keyed by (weight, vertex, parent) - which must agree on the weight; (2) the set of
odd-degree vertices of that tree, whose cardinality is even by the handshake lemma; (3) a
minimum-weight perfect matching on those vertices, computed twice: by plain recursion on the
sorted vertex list and by a bitmask dynamic program, both exhaustive, so optimality is exact
rather than heuristic (the odd set never exceeds twelve vertices here, i.e. at most 10395
matchings); (4) the union of the tree and the matching, a connected multigraph with every degree
even, traversed by an edge-id-aware iterative Hierholzer pass to get the Eulerian circuit, which is
then shortcut - repeated vertices are skipped - into a Hamiltonian cycle.

Checks: the two MST engines agree on the weight and the tree is spanning and acyclic; the two
matching engines agree on the matched weight; the multigraph has only even degrees, its Eulerian
circuit consumes every edge id exactly once, visits each vertex degree/2 times and returns to its
start; the shortcut tour is a permutation whose length is at most the walk length
(MST + matching, since metric shortcutting never lengthens) and therefore at most 1.5 times the
optimum once the provable matching <= optimum/2 is used - note the factor 3/2 bounds the tour
against the *optimum*, not against the MST, which the square-with-centre instance illustrates
(MST = 2*sqrt(2), tour = 3 + sqrt(2) = 4.414 > 1.5 * MST = 4.243). Hand-checked instances: the
unit square (three unit sides plus the missing side as the matching, tour 4 = optimum), the square
with a centre point (MST = 2*sqrt(2), the four corners odd, matching 2, Eulerian walk 2 + 2*sqrt(2),
shortcut tour exactly 3 + sqrt(2), which is also the brute-force optimum), a triangle and
collinear cities where the construction is forced to be optimal (perimeter, and 2 * range). Then
ten LCG instances (n = 8) plus a 12-city instance: optimum <= christofides, the 3/2 factor against
the optimum, MST <= optimum, the MST tie-break dependence, the guarantee under relabelling of the
input order, and determinism of repeated runs.
"""
import heapq
import itertools
import math

INF = float("inf")


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def weight_matrix(points):
    n = len(points)
    return [[dist(points[i], points[j]) for j in range(n)] for i in range(n)]


def kruskal(n, w):
    """Returns (weight, edge list) of a minimum spanning tree."""
    edges = sorted((w[i][j], i, j) for i in range(n) for j in range(i + 1, n))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0.0
    tree = []
    for (cost, i, j) in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append((i, j))
            total += cost
    return total, tree


def prim(n, w):
    """Independent MST: Prim with a heap keyed by (weight, vertex, parent)."""
    if n == 0:
        return 0.0, []
    visited = [False] * n
    visited[0] = True
    heap = [(w[0][j], j, 0) for j in range(1, n)]
    heapq.heapify(heap)
    total = 0.0
    tree = []
    while heap:
        (cost, v, u) = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        total += cost
        tree.append((u, v))
        for x in range(n):
            if not visited[x]:
                heapq.heappush(heap, (w[v][x], x, v))
    return total, tree


def degrees(n, edges):
    deg = [0] * n
    for (u, v) in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def traversed_edges(circuit):
    """The unordered edges walked by a vertex sequence, sorted - for race-free comparison."""
    return sorted(tuple(sorted((circuit[i], circuit[i + 1]))) for i in range(len(circuit) - 1))


def min_matching_recursive(vertices, w):
    """Exhaustive minimum-weight perfect matching (i, j) pairs, plain recursion."""
    verts = sorted(vertices)
    if not verts:
        return 0.0, []

    def solve(remaining):
        if not remaining:
            return 0.0, []
        first = remaining[0]
        best = (INF, None)
        for index in range(1, len(remaining)):
            other = remaining[index]
            rest = remaining[1:index] + remaining[index + 1:]
            sub_cost, sub_pairs = solve(rest)
            total = w[first][other] + sub_cost
            if total < best[0] - 1e-12:
                best = (total, [(first, other)] + sub_pairs)
        return best

    return solve(tuple(verts))


def min_matching_bitmask(vertices, w):
    """Independent exhaustive matcher: bitmask DP over the odd set."""
    verts = sorted(vertices)
    m = len(verts)
    if m == 0:
        return 0.0, []
    full = (1 << m) - 1
    best = {0: 0.0}
    choice = {}
    for mask in range(full + 1):
        if mask not in best:
            continue
        remaining = [k for k in range(m) if not (mask >> k) & 1]
        if not remaining:
            continue
        first = remaining[0]
        for other in remaining[1:]:
            nxt = mask | (1 << first) | (1 << other)
            candidate = best[mask] + w[verts[first]][verts[other]]
            if nxt not in best or candidate < best[nxt] - 1e-12:
                best[nxt] = candidate
                choice[nxt] = (mask, verts[first], verts[other])
    pairs = []
    mask = full
    while mask:
        prev, a, b = choice[mask]
        pairs.append((a, b))
        mask = prev
    return best[full], sorted(pairs)


def euler_circuit(n, edge_list):
    """Iterative Hierholzer on an undirected multigraph; returns the vertex sequence."""
    adjacency = [[] for _ in range(n)]
    for eid, (u, v) in enumerate(edge_list):
        adjacency[u].append((v, eid))
        adjacency[v].append((u, eid))
    for v in range(n):
        adjacency[v].sort()
    used = [False] * len(edge_list)
    pointer = [0] * n
    start = edge_list[0][0]
    stack = [start]
    circuit = []
    while stack:
        v = stack[-1]
        while pointer[v] < len(adjacency[v]) and used[adjacency[v][pointer[v]][1]]:
            pointer[v] += 1
        if pointer[v] == len(adjacency[v]):
            circuit.append(stack.pop())
        else:
            (u, eid) = adjacency[v][pointer[v]]
            used[eid] = True
            stack.append(u)
    return circuit[::-1]


def christofides(points, tree=None):
    """Full Christofides construction. Returns a dict with the tour and every intermediate.

    ``tree`` lets the caller substitute a specific spanning tree, which is how the file shows
    that the achieved length depends on the MST tie-breaking while staying inside the 3/2 bound.
    """
    n = len(points)
    if n <= 1:
        return {"tour": list(range(n)), "cycle": list(range(n)), "mst_cost": 0.0,
                "kruskal_cost": 0.0, "prim_cost": 0.0, "tree": [], "prim_tree": [], "odd": [],
                "matching": [], "match_cost": 0.0, "dp_cost": 0.0, "dp_matching": [],
                "multigraph": [], "circuit": list(range(n))}
    w = weight_matrix(points)
    kruskal_cost, kruskal_tree = kruskal(n, w)
    prim_cost, prim_tree = prim(n, w)
    if tree is None:
        tree = kruskal_tree
    mst_cost = sum(w[u][v] for (u, v) in tree)
    deg = degrees(n, tree)
    odd = sorted(v for v in range(n) if deg[v] % 2 == 1)
    match_cost, matching = min_matching_recursive(odd, w)
    dp_cost, dp_matching = min_matching_bitmask(odd, w)
    multigraph = list(tree) + list(matching)
    circuit = euler_circuit(n, multigraph)
    tour = []
    seen = set()
    for v in circuit:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    tour.append(circuit[0])
    return {
        "tour": tour[:-1],
        "cycle": tour,
        "mst_cost": mst_cost,
        "kruskal_cost": kruskal_cost,
        "prim_cost": prim_cost,
        "tree": tree,
        "prim_tree": prim_tree,
        "odd": odd,
        "matching": matching,
        "match_cost": match_cost,
        "dp_cost": dp_cost,
        "dp_matching": dp_matching,
        "multigraph": multigraph,
        "circuit": circuit,
    }


def tour_cost(w, tour):
    n = len(tour)
    return sum(w[tour[i]][tour[(i + 1) % n]] for i in range(n))


def brute_force_optimal(w):
    n = len(w)
    if n == 1:
        return 0.0
    best = INF
    for perm in itertools.permutations(range(1, n)):
        best = min(best, tour_cost(w, (0,) + perm))
    return best


def doubled_tree_cost(points):
    """The simpler 2-approximation: walk the doubled MST and shortcut it."""
    n = len(points)
    w = weight_matrix(points)
    mst_cost, tree = kruskal(n, w)
    multigraph = list(tree) + list(tree)
    circuit = euler_circuit(n, multigraph)
    tour = []
    seen = set()
    for v in circuit:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    return tour_cost(w, tour)


def lcg_points(n, seed, limit=30):
    points = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        x = seed % limit
        seed = (seed * 75 + 74) % 65537
        y = seed % limit
        points.append((float(x), float(y)))
    return points


def main():
    # --- The unit square: three sides plus the missing side as matching ------------------
    square = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    result = christofides(square)
    w = weight_matrix(square)
    assert abs(result["mst_cost"] - 3.0) <= 1e-12, "three unit edges span the square"
    assert abs(result["kruskal_cost"] - result["prim_cost"]) <= 1e-12, "Prim and Kruskal agree"
    assert result["tree"] == [(0, 1), (0, 2), (1, 3)], "the documented Kruskal tie-breaking"
    assert result["odd"] == [2, 3], "the two leaves of the V-shaped tree"
    assert abs(result["match_cost"] - 1.0) <= 1e-12, "the missing side connects them"
    assert abs(result["dp_cost"] - result["match_cost"]) <= 1e-12
    assert abs(tour_cost(w, result["tour"]) - 4.0) <= 1e-9, "here it hits the optimum 4"
    assert sorted(result["tour"]) == [0, 1, 2, 3], "the tour is a permutation"
    assert len(result["circuit"]) == len(result["multigraph"]) + 1
    assert result["circuit"][0] == result["circuit"][-1], "the Eulerian walk is closed"
    mult_deg = degrees(4, result["multigraph"])
    assert traversed_edges(result["circuit"]) == sorted(tuple(sorted(e))
                                                        for e in result["multigraph"]), \
        "the Eulerian circuit traverses every multigraph edge exactly once"
    for v in range(4):
        assert result["circuit"][:-1].count(v) * 2 == mult_deg[v], \
            "a vertex occurs degree/2 times in a closed Eulerian walk"
    optimum_square = brute_force_optimal(w)
    assert abs(optimum_square - 4.0) <= 1e-12
    assert tour_cost(w, result["tour"]) <= result["mst_cost"] + result["match_cost"] + 1e-12
    assert tour_cost(w, result["tour"]) <= 1.5 * optimum_square + 1e-12
    assert result["match_cost"] <= 0.5 * optimum_square + 1e-12

    # Any other minimum spanning tree here must behave the same way (all of them are a square
    # minus one side, whose two endpoints form the odd pair), so the achieved length is 4 again.
    alt = christofides(square, tree=[(0, 1), (1, 3), (2, 3)])
    assert abs(sum(w[u][v] for (u, v) in alt["tree"]) - 3.0) <= 1e-12, "same MST weight"
    assert alt["odd"] == [0, 2]
    assert abs(tour_cost(w, alt["tour"]) - 4.0) <= 1e-9
    assert sorted(alt["tour"]) == [0, 1, 2, 3]

    # --- Square plus a centre point: the classic 3 + sqrt(2) value ----------------------
    centred = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
    res = christofides(centred)
    w_c = weight_matrix(centred)
    assert abs(res["mst_cost"] - 2.0 * math.sqrt(2.0)) <= 1e-12, "a four-armed star to the centre"
    assert res["odd"] == [0, 1, 2, 3], "the centre has degree 4, every corner degree 1"
    assert abs(res["match_cost"] - 2.0) <= 1e-12, "two unit sides beat the two diagonals"
    assert abs(res["dp_cost"] - res["match_cost"]) <= 1e-12
    walk = res["mst_cost"] + res["match_cost"]
    assert abs(walk - (2.0 + 2.0 * math.sqrt(2.0))) <= 1e-12
    assert abs(tour_cost(w_c, res["tour"]) - (3.0 + math.sqrt(2.0))) <= 1e-9, "the classic value"
    optimum_centred = brute_force_optimal(w_c)
    assert abs(optimum_centred - (3.0 + math.sqrt(2.0))) <= 1e-9, "and it is optimal here"
    assert tour_cost(w_c, res["tour"]) <= walk + 1e-12, "metric shortcutting never lengthens"
    assert tour_cost(w_c, res["tour"]) <= 1.5 * optimum_centred + 1e-12
    assert 1.5 * res["mst_cost"] < tour_cost(w_c, res["tour"]) - 1e-9, \
        "the 3/2 factor is measured against the optimum, not against the MST"

    # --- Degenerate sizes ---------------------------------------------------------------
    for n in (1, 2):
        tiny = lcg_points(n, 3)
        res = christofides(tiny)
        assert sorted(res["tour"]) == list(range(n))
        if n == 2:
            assert abs(tour_cost(weight_matrix(tiny), res["tour"])
                       - 2.0 * dist(tiny[0], tiny[1])) <= 1e-12
        else:
            assert res["tour"] == [0]

    # --- Triangle and collinear cities: Christofides is forced to be optimal ------------
    triangle = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
    res = christofides(triangle)
    w_tri = weight_matrix(triangle)
    assert abs(tour_cost(w_tri, res["tour"]) - 12.0) <= 1e-12, "3 + 4 + 5 = 12"
    assert abs(brute_force_optimal(w_tri) - 12.0) <= 1e-12
    line = [(float(3 * i), 0.0) for i in range(6)]
    res = christofides(line)
    w_line = weight_matrix(line)
    optimum = brute_force_optimal(w_line)
    assert abs(optimum - 30.0) <= 1e-9, "closed walk on a line: 2 * 15"
    assert abs(tour_cost(w_line, res["tour"]) - optimum) <= 1e-9
    assert res["odd"] == [0, 5], "only the two ends have odd degree in the path MST"

    # --- Random LCG instances against brute force --------------------------------------
    seed = 31415
    for trial in range(10):
        n = 8
        points = lcg_points(n, seed)
        seed = (seed * 75 + 74) % 65537
        w = weight_matrix(points)
        for i in range(n):                         # the instance must be metric
            for j in range(n):
                assert w[i][j] == w[j][i]
                for k in range(n):
                    assert w[i][k] <= w[i][j] + w[j][k] + 1e-12, "triangle inequality"
        res = christofides(points)
        optimum = brute_force_optimal(w)
        cost = tour_cost(w, res["tour"])
        assert sorted(res["tour"]) == list(range(n)), "tour is a permutation"
        assert abs(res["mst_cost"] - res["prim_cost"]) <= 1e-9
        assert len(res["tree"]) == n - 1 and len(res["prim_tree"]) == n - 1, "spanning trees"
        tree_adjacency = [[] for _ in range(n)]
        for (u, v) in res["tree"]:
            tree_adjacency[u].append(v)
            tree_adjacency[v].append(u)
        seen = {0}
        stack = [0]
        while stack:
            v = stack.pop()
            for w in tree_adjacency[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        assert len(seen) == n and len(res["tree"]) == n - 1, \
            "a connected graph on n vertices with n - 1 edges is a spanning tree"
        assert abs(res["match_cost"] - res["dp_cost"]) <= 1e-9, "the two matchers agree"
        assert len(res["odd"]) % 2 == 0, "handshake lemma: an even number of odd vertices"
        assert len(res["matching"]) == len(res["odd"]) // 2
        assert sorted(v for pair in res["matching"] for v in pair) == res["odd"]
        deg = degrees(n, res["multigraph"])
        assert all(d % 2 == 0 for d in deg), "tree + matching is Eulerian"
        assert len(res["multigraph"]) == (n - 1) + len(res["odd"]) // 2
        assert optimum <= cost + 1e-9, "no tour can beat the optimum"
        assert cost <= res["mst_cost"] + res["match_cost"] + 1e-9, "shortcut of the Eulerian walk"
        assert res["match_cost"] <= 0.5 * optimum + 1e-9, "matching <= half the optimal tour"
        assert cost <= 1.5 * optimum + 1e-9, "the 3/2 factor"
        assert res["mst_cost"] <= optimum + 1e-9, "the MST lower-bounds the optimum"
        # Relabelling the cities keeps the instance (and its optimum) but may change the
        # tie-breaking of Kruskal and Prim; the achieved length must still satisfy the bound.
        relabelled = [points[(i + 3) % n] for i in range(n)]
        relabelled_cost = tour_cost(weight_matrix(relabelled),
                                    christofides(relabelled)["tour"])
        assert optimum <= relabelled_cost + 1e-9
        assert relabelled_cost <= 1.5 * optimum + 1e-9, "the guarantee is labelling-independent"
        repeat = christofides(points)
        assert repeat["tour"] == res["tour"] and repeat["circuit"] == res["circuit"], \
            "deterministic"

    # --- A larger, matching-heavy instance --------------------------------------------
    big = lcg_points(12, 999, limit=50)
    res = christofides(big)
    w_big = weight_matrix(big)
    cost = tour_cost(w_big, res["tour"])
    assert sorted(res["tour"]) == list(range(12))
    assert len(res["odd"]) >= 2 and len(res["odd"]) % 2 == 0
    assert cost <= res["mst_cost"] + res["match_cost"] + 1e-9
    assert cost >= res["mst_cost"] - 1e-9
    assert all(d % 2 == 0 for d in degrees(12, res["multigraph"]))
    assert traversed_edges(res["circuit"]) == sorted(tuple(sorted(e)) for e in res["multigraph"])
    assert len(set(res["tour"])) == 12
    # The doubled-tree 2-approximation is never better by more than the matching saving.
    assert cost <= doubled_tree_cost(big) + 1e-9


if __name__ == "__main__":
    main()
    print("PASS")
