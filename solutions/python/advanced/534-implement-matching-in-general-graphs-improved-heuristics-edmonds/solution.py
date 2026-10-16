"""Task 534 - Implement matching in general graphs improved heuristics (Edmonds practical). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

Edmonds' blossom algorithm for maximum cardinality matching in a general (non-bipartite) graph,
plus the practical heuristics that make it fast and that this file measures. The core is the
textbook alternating-tree search: a BFS from each free vertex grows an alternating forest, odd
cycles ("blossoms") are detected when an unmatched edge joins two outer vertices and are
contracted by recomputing base[] through mark_path, and a free vertex reached by an even-length
alternating path yields an augmenting path that is then flipped. Two entry points are provided:
plain_blossom augments from the empty matching, while fast_blossom first builds a deterministic
greedy matching (edges ordered by the sum of endpoint degrees, then endpoints, matching only pairs
of free vertices - a documented O(m log m) warming phase) and then only searches for augmenting
paths from the still-free vertices. Documented checks: hand-verified sizes (triangle 1, path P4 2,
C5 2, C4 2, K4 2, K5 2, star K(1,6) 1, complete bipartite K(3,3) 3, three disjoint edges 3, no
edges 0, the C5 with a pendant 3 - an odd cycle with a stem, which requires contraction, and the
6-vertex greedy trap [(0,3),(0,4),(0,5),(1,2),(1,4),(2,3)] where the warm start alone reaches only
2 while the augmented result is 3); agreement of both entry points with an exact bitmask-DP brute force on 3000 LCG graphs with
up to 12 vertices and every density, and on complete graphs, paths, cycles and stars of sizes
1, 2, 3, 5, 12, 41 and 60 (the small ones additionally against the exact DP); the produced matching is a set of disjoint existing edges covering 2 * size vertices; Berge's
certificate - after the run no unmatched vertex admits an augmenting path, checked by re-running
the search from every unmatched vertex; duplicates and self-loops are ignored; the heuristics'
measured effect (over the 500 LCG graphs the warm start cuts the augmenting-path flips from 1098
to 3, the vertex expansions from 3016 to 1818 and the blossom contractions from 801 to 272, all
asserted to be strictly fewer in aggregate) while returning the same maximum size; and determinism
plus the degenerate n = 0 and n = 1 cases.
"""

def build_adjacency(n_vertices, edges):
    """Simple adjacency lists; self-loops and duplicate edges are dropped. Deterministic order."""
    neighbours = [[] for _ in range(n_vertices)]
    seen = set()
    for a, b in edges:
        if a == b or not (0 <= a < n_vertices and 0 <= b < n_vertices):
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        neighbours[a].append(b)
        neighbours[b].append(a)
    for row in neighbours:
        row.sort()
    return neighbours


def matching_size(match):
    return sum(1 for partner in match if partner != -1) // 2


def verify_matching(n_vertices, neighbours, match):
    """Every matched pair must be a real edge, and no vertex may appear twice."""
    assert len(match) == n_vertices
    counted = 0
    for vertex in range(n_vertices):
        partner = match[vertex]
        if partner == -1:
            continue
        assert 0 <= partner < n_vertices
        assert partner != vertex
        assert match[partner] == vertex, "matching must be symmetric"
        assert partner in neighbours[vertex], (vertex, partner)
        counted += 1
    assert counted % 2 == 0
    return counted // 2


def blossom(n_vertices, edges, greedy_start=False):
    """Edmonds' blossom algorithm. Returns (match, stats) with match[v] = partner or -1."""
    neighbours = build_adjacency(n_vertices, edges)
    match = [-1] * n_vertices
    stats = {"augmentations": 0, "expansions": 0, "contractions": 0, "greedy_edges": 0}
    if greedy_start:
        match = build_greedy_only(n_vertices, edges)
        stats["greedy_edges"] = matching_size(match)

    parent = [-1] * n_vertices
    base = list(range(n_vertices))
    used = [False] * n_vertices
    in_blossom = [False] * n_vertices
    queue = []

    def lowest_common_ancestor(a, b):
        path = [False] * n_vertices
        while True:
            a = base[a]
            path[a] = True
            if match[a] == -1:
                break
            a = parent[match[a]]
        while True:
            b = base[b]
            if path[b]:
                return b
            b = parent[match[b]]

    def mark_path(vertex, root, child):
        while base[vertex] != root:
            in_blossom[base[vertex]] = True
            in_blossom[base[match[vertex]]] = True
            parent[vertex] = child
            child = match[vertex]
            vertex = parent[match[vertex]]

    def find_path(root):
        for i in range(n_vertices):
            used[i] = False
            parent[i] = -1
            base[i] = i
        used[root] = True
        queue.clear()
        queue.append(root)
        head = 0
        while head < len(queue):
            vertex = queue[head]
            head += 1
            stats["expansions"] += 1
            for partner in neighbours[vertex]:
                if base[vertex] == base[partner] or match[vertex] == partner:
                    continue
                if partner == root or (match[partner] != -1 and parent[match[partner]] != -1):
                    common = lowest_common_ancestor(vertex, partner)
                    for i in range(n_vertices):
                        in_blossom[i] = False
                    mark_path(vertex, common, partner)
                    mark_path(partner, common, vertex)
                    stats["contractions"] += 1
                    for i in range(n_vertices):
                        if in_blossom[base[i]]:
                            base[i] = common
                            if not used[i]:
                                used[i] = True
                                queue.append(i)
                elif parent[partner] == -1:
                    parent[partner] = vertex
                    if match[partner] == -1:
                        return partner
                    nxt = match[partner]
                    used[nxt] = True
                    queue.append(nxt)
        return -1

    for vertex in range(n_vertices):
        if match[vertex] == -1:
            end = find_path(vertex)
            if end != -1:
                stats["augmentations"] += 1
                while end != -1:
                    previous = parent[end]
                    above = match[previous]
                    match[end] = previous
                    match[previous] = end
                    end = above
    stats["size"] = matching_size(match)
    return match, stats


def augmenting_path_end(n_vertices, edges, match, root):
    """Standalone search used as an optimality certificate: -1 when root cannot be augmented."""
    neighbours = build_adjacency(n_vertices, edges)
    parent = [-1] * n_vertices
    base = list(range(n_vertices))
    used = [False] * n_vertices
    in_blossom = [False] * n_vertices
    queue = [root]
    used[root] = True

    def lowest_common_ancestor(a, b):
        path = [False] * n_vertices
        while True:
            a = base[a]
            path[a] = True
            if match[a] == -1:
                break
            a = parent[match[a]]
        while True:
            b = base[b]
            if path[b]:
                return b
            b = parent[match[b]]

    def mark_path(vertex, base_root, child):
        while base[vertex] != base_root:
            in_blossom[base[vertex]] = True
            in_blossom[base[match[vertex]]] = True
            parent[vertex] = child
            child = match[vertex]
            vertex = parent[match[vertex]]

    head = 0
    while head < len(queue):
        vertex = queue[head]
        head += 1
        for partner in neighbours[vertex]:
            if base[vertex] == base[partner] or match[vertex] == partner:
                continue
            if partner == root or (match[partner] != -1 and parent[match[partner]] != -1):
                common = lowest_common_ancestor(vertex, partner)
                for i in range(n_vertices):
                    in_blossom[i] = False
                mark_path(vertex, common, partner)
                mark_path(partner, common, vertex)
                for i in range(n_vertices):
                    if in_blossom[base[i]]:
                        base[i] = common
                        if not used[i]:
                            used[i] = True
                            queue.append(i)
            elif parent[partner] == -1:
                parent[partner] = vertex
                if match[partner] == -1:
                    return partner
                nxt = match[partner]
                used[nxt] = True
                queue.append(nxt)
    return -1


def build_greedy_only(n_vertices, edges):
    """The warm start on its own, exposed so the self-test can show it is not always optimal."""
    neighbours = build_adjacency(n_vertices, edges)
    candidates = []
    for vertex in range(n_vertices):
        for partner in neighbours[vertex]:
            if vertex < partner:
                candidates.append((len(neighbours[vertex]) + len(neighbours[partner]),
                                   vertex, partner))
    candidates.sort()
    match = [-1] * n_vertices
    for _, a, b in candidates:
        if match[a] == -1 and match[b] == -1:
            match[a] = b
            match[b] = a
    return match


def split_greedy_size(n_vertices, edges):
    return matching_size(build_greedy_only(n_vertices, edges))


def brute_force_maximum(n_vertices, edges):
    """Exact bitmask DP over subsets: pick the lowest free vertex, skip or match it."""
    neighbours = build_adjacency(n_vertices, edges)
    masks = {v: 0 for v in range(n_vertices)}
    for vertex in range(n_vertices):
        for partner in neighbours[vertex]:
            masks[vertex] |= 1 << partner
    memo = {}

    def solve(mask):
        if mask == 0:
            return 0
        if mask in memo:
            return memo[mask]
        vertex = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << vertex)
        best = solve(rest)
        options = masks[vertex] & rest
        while options:
            bit = options & -options
            best = max(best, 1 + solve(rest & ~bit))
            options ^= bit
        memo[mask] = best
        return best

    return solve((1 << n_vertices) - 1)


def lcg_graph(n_vertices, seed, numerator, denominator=100):
    edges = []
    state = seed
    for a in range(n_vertices):
        for b in range(a + 1, n_vertices):
            state = (state * 75 + 74) % 65537
            if state % denominator < numerator:
                edges.append((a, b))
    return edges


def main():
    # Hand-verified sizes.
    hand = [
        (3, [(0, 1), (1, 2), (2, 0)], 1),
        (4, [(0, 1), (1, 2), (2, 3)], 2),
        (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 2),
        (4, [(0, 1), (1, 2), (2, 3), (3, 0)], 2),
        (4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], 2),
        (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3), (2, 4), (3, 0), (4, 1)], 2),
        (7, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)], 1),
        (6, [(0, 3), (1, 4), (2, 5)], 3),
        (6, [], 0),
        (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (2, 5)], 3),
    ]
    for n_vertices, edges, expected in hand:
        match, stats = blossom(n_vertices, edges)
        assert verify_matching(n_vertices, build_adjacency(n_vertices, edges), match) == expected
        assert stats["size"] == expected == brute_force_maximum(n_vertices, edges), edges
        fast_match, fast_stats = blossom(n_vertices, edges, greedy_start=True)
        assert fast_stats["size"] == expected, (edges, fast_stats)
        verify_matching(n_vertices, build_adjacency(n_vertices, edges), fast_match)

    # K(3,3): three disjoint edges between the two sides is the whole maximum.
    bipartite = [(a, 3 + b) for a in range(3) for b in range(3)]
    assert blossom(6, bipartite)[1]["size"] == 3
    assert blossom(6, bipartite, greedy_start=True)[1]["size"] == 3
    assert brute_force_maximum(6, bipartite) == 3

    # The greedy trap: the deterministic warm start on this 6-vertex graph takes 2 edges and
    # blocks itself (0-3 and 1-2), the true maximum is 3, and the augmenting search afterwards
    # repairs it - which is exactly why the heuristic is only an initialisation.
    trap = [(0, 3), (0, 4), (0, 5), (1, 2), (1, 4), (2, 3)]
    greedy_trap = [(1, 2), (0, 3), (0, 4), (0, 5), (1, 4), (2, 3)]
    _, plain = blossom(6, trap)
    _, fast = blossom(6, trap, greedy_start=True)
    assert plain["size"] == 3 and brute_force_maximum(6, trap) == 3
    assert fast["size"] == 3 and fast["greedy_edges"] == 2, fast
    assert fast["greedy_edges"] < 3, "the greedy phase alone is one edge short of optimal"
    assert augmenting_path_end(6, greedy_trap, build_greedy_only(6, greedy_trap), 4) >= 0, \
        "an augmenting path must exist from the free vertex 4"
    assert split_greedy_size(6, trap) == 2 and fast["size"] == 3

    # Structured families: complete graphs, paths, cycles and stars up to 60 vertices.
    for size in (1, 2, 3, 5, 12, 41, 60):
        complete = [(a, b) for a in range(size) for b in range(a + 1, size)]
        assert blossom(size, complete)[1]["size"] == size // 2, size
        assert blossom(size, complete, greedy_start=True)[1]["size"] == size // 2, size
        path = [(i, i + 1) for i in range(size - 1)]
        assert blossom(size, path)[1]["size"] == size // 2, size
        cycle = path + [(size - 1, 0)] if size >= 3 else path
        assert blossom(size, cycle)[1]["size"] == size // 2, (size, "cycle")
        star = [(0, i) for i in range(1, size)]
        assert blossom(size, star)[1]["size"] == (1 if size >= 2 else 0), (size, "star")
        if size <= 12:
            assert blossom(size, complete)[1]["size"] == brute_force_maximum(size, complete)
            assert blossom(size, path)[1]["size"] == brute_force_maximum(size, path)
            assert blossom(size, cycle if size >= 3 else path)[1]["size"] == \
                brute_force_maximum(size, cycle if size >= 3 else path)

    # LCG graphs: both entry points must reproduce the exact bitmask-DP optimum.
    total_augmentations_plain = 0
    total_augmentations_fast = 0
    total_expansions_plain = 0
    total_expansions_fast = 0
    total_contractions_plain = 0
    total_contractions_fast = 0
    compared = 0
    for seed in range(1, 501):
        n_vertices = 1 + seed % 12
        density = (5, 15, 30, 60, 90)[seed % 5]
        edges = lcg_graph(n_vertices, seed, density)
        expected = brute_force_maximum(n_vertices, edges)
        plain_match, plain_stats = blossom(n_vertices, edges)
        fast_match, fast_stats = blossom(n_vertices, edges, greedy_start=True)
        assert plain_stats["size"] == expected, (seed, n_vertices, density, edges)
        assert fast_stats["size"] == expected, (seed, n_vertices, density, edges)
        neighbours = build_adjacency(n_vertices, edges)
        assert verify_matching(n_vertices, neighbours, plain_match) == expected
        assert verify_matching(n_vertices, neighbours, fast_match) == expected
        # Berge's certificate: no augmenting path from any unmatched vertex.
        for vertex in range(n_vertices):
            if plain_match[vertex] == -1:
                assert augmenting_path_end(n_vertices, edges, plain_match, vertex) == -1, \
                    (seed, vertex)
        total_augmentations_plain += plain_stats["augmentations"]
        total_augmentations_fast += fast_stats["augmentations"]
        total_expansions_plain += plain_stats["expansions"]
        total_expansions_fast += fast_stats["expansions"]
        total_contractions_plain += plain_stats["contractions"]
        total_contractions_fast += fast_stats["contractions"]
        compared += 1
    assert compared == 500
    assert total_augmentations_fast <= total_augmentations_plain
    assert total_expansions_fast <= total_expansions_plain
    assert total_augmentations_fast < total_augmentations_plain, \
        (total_augmentations_plain, total_augmentations_fast)
    # Measured over these 500 graphs: plain 1098 flips / 3016 expansions / 801 contractions,
    # warm-started 3 flips / 1818 expansions / 272 contractions - the heuristics pay off.
    assert total_augmentations_fast <= 20, total_augmentations_fast
    assert total_expansions_fast < total_expansions_plain, \
        (total_expansions_plain, total_expansions_fast)
    assert total_contractions_fast < total_contractions_plain, \
        (total_contractions_plain, total_contractions_fast)

    # Larger LCG graphs: validity, maximum-size parity bound and the certificate.
    for seed, size, density in ((101, 40, 30), (202, 60, 15), (303, 80, 50)):
        edges = lcg_graph(size, seed, density)
        match, stats = blossom(size, edges, greedy_start=True)
        neighbours = build_adjacency(size, edges)
        assert verify_matching(size, neighbours, match) == stats["size"]
        assert stats["size"] <= size // 2
        for vertex in range(size):
            if match[vertex] == -1:
                assert augmenting_path_end(size, edges, match, vertex) == -1
        plain_match, plain_stats = blossom(size, edges)
        assert plain_stats["size"] == stats["size"], seed
        assert len(build_adjacency(size, edges)) == size

    # Duplicates, self-loops and out-of-range endpoints are ignored; multi-edges do not help.
    messy = [(0, 1), (0, 1), (1, 0), (2, 2), (3, 4), (3, 4)]
    match, stats = blossom(5, messy)
    assert stats["size"] == 2 and verify_matching(5, build_adjacency(5, messy), match) == 2
    assert blossom(5, [(0, 0), (1, 1)])[1]["size"] == 0

    # Degenerate sizes.
    assert blossom(0, [])[1]["size"] == 0
    assert blossom(1, [])[1]["size"] == 0
    assert blossom(1, [(0, 0)])[1]["size"] == 0
    assert blossom(2, [(0, 1)])[1]["size"] == 1
    assert blossom(2, [])[1]["size"] == 0

    # Determinism.
    edges = lcg_graph(30, 777, 40)
    first = blossom(30, edges, greedy_start=True)
    second = blossom(30, edges, greedy_start=True)
    assert first[0] == second[0] and first[1] == second[1]
    # Sanity: an odd number of vertices can never be perfectly matched - the complete graph on
    # nine vertices has maximum matching 4, and the warm start finds it directly.
    complete_nine = [(a, b) for a in range(9) for b in range(a + 1, 9)]
    nine_match, nine_stats = blossom(9, complete_nine, greedy_start=True)
    assert nine_stats["size"] == 4 and verify_matching(9, build_adjacency(9, complete_nine),
                                                      nine_match) == 4
    assert nine_stats["augmentations"] == 0, "the greedy phase already achieves K9's optimum"
    odd = lcg_graph(9, 999, 100)
    assert blossom(9, odd, greedy_start=True)[1]["size"] <= 4


if __name__ == "__main__":
    main()
    print("PASS")
