"""Task 343 - Graph isomorphism: simple checks with degree-sequence invariants. (Tier: Intermediate, CTCI Ch 4)

These are necessary conditions, deliberately not sufficient:
  * identical sorted degree sequences;
  * identical triangle counts;
  * identical connected-component size multisets.
Isomorphic relabelings pass all three.  C6 vs C3+C3 (two disjoint triangles) has
the same degree sequence [2]*6 but fails on the component multiset {6} vs {3,3};
K3,3 vs the triangular prism matches degrees and components but fails on
triangles (0 vs 2).  Documented counterexample for the "not sufficient" part:
the cube Q3 and the Moebius ladder V8 both pass all three checks yet are not
isomorphic - the cube is bipartite and V8 is not (asserted below, along with the
diameters 3 vs 2).  A vertex-refinement step would be needed to separate more
such pairs.
"""


def degrees(adj):
    return sorted(len(set(neighbours)) for neighbours in adj)


def triangle_count(adj):
    sets = [set(neighbours) for neighbours in adj]
    count = 0
    for u in range(len(adj)):
        for v in sets[u]:
            if v <= u:
                continue
            for w in sets[u] & sets[v]:
                if w > v:
                    count += 1
    return count


def component_sizes(adj):
    """Component size multiset via iterative DFS."""
    seen = [False] * len(adj)
    sizes = []
    for start in range(len(adj)):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        sizes.append(size)
    return sorted(sizes)


def component_sizes_union_find(adj):
    """Independent oracle for the component multiset."""
    parent = list(range(len(adj)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u in range(len(adj)):
        for v in set(adj[u]):
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                parent[root_u] = root_v
    counts = {}
    for i in range(len(adj)):
        root = find(i)
        counts[root] = counts.get(root, 0) + 1
    return sorted(counts.values())


def check_isomorphic(a, b):
    """Necessary-condition checker: never proves isomorphism, but rules it out."""
    if len(a) != len(b):
        return False
    if degrees(a) != degrees(b):
        return False
    if triangle_count(a) != triangle_count(b):
        return False
    if component_sizes(a) != component_sizes(b):
        return False
    return True


def diameter(adj):
    """BFS from every vertex; -1 when the graph is disconnected."""
    best = 0
    for start in range(len(adj)):
        dist = [-1] * len(adj)
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if any(d < 0 for d in dist):
            return -1
        best = max(best, max(dist))
    return best


def relabel(adj, perm):
    """perm maps old vertex -> new vertex; returns the relabelled adjacency."""
    n = len(adj)
    out = [[] for _ in range(n)]
    for u in range(n):
        out[perm[u]] = sorted(perm[v] for v in set(adj[u]))
    return out


def cycle_graph(n):
    return [[(i - 1) % n, (i + 1) % n] for i in range(n)]


def two_triangles():
    return [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]


def k33():
    return [[3, 4, 5], [3, 4, 5], [3, 4, 5], [0, 1, 2], [0, 1, 2], [0, 1, 2]]


def prism():
    return [[1, 2, 3], [0, 2, 4], [0, 1, 5], [4, 5, 0], [3, 5, 1], [3, 4, 2]]


def cube_q3():
    return [[i ^ (1 << bit) for bit in range(3)] for i in range(8)]


def mobius_v8():
    """Moebius ladder on 8 vertices: the C8 plus the four opposite rungs."""
    adj = [set() for _ in range(8)]
    for i in range(8):
        adj[i].add((i + 1) % 8)
        adj[(i + 1) % 8].add(i)
    for i in range(4):
        adj[i].add(i + 4)
        adj[i + 4].add(i)
    return [sorted(neighbours) for neighbours in adj]


def is_bipartite(adj):
    """Cheap refinement invariant: BFS 2-colouring."""
    color = [-1] * len(adj)
    for start in range(len(adj)):
        if color[start] >= 0:
            continue
        color[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if color[v] < 0:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def lcg_graphs(count, n=7, seed=8009):
    graphs = []
    for _ in range(count):
        adj = [[] for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                seed = (seed * 75 + 74) % 65537
                if seed % 2 == 0:
                    adj[u].append(v)
                    adj[v].append(u)
        graphs.append(adj)
    return graphs


def lcg_permutation(n, seed):
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        seed = (seed * 75 + 74) % 65537
        j = seed % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm, seed


def main():
    c5 = cycle_graph(5)
    c6 = cycle_graph(6)
    path4 = [[1], [0, 2], [1, 3], [2]]
    star4 = [[1, 2, 3, 4], [0], [0], [0], [0]]
    bowtie = [[1, 2, 3, 4], [0, 2], [0, 1], [0, 4], [0, 3]]

    # isomorphic pairs: relabelings pass
    assert check_isomorphic(c5, c5)
    pentagram = relabel(c5, [0, 2, 4, 1, 3])           # star polygon relabelling
    assert pentagram == [[2, 3], [3, 4], [0, 4], [0, 1], [1, 2]]
    assert pentagram != c5
    assert check_isomorphic(c5, pentagram)
    assert check_isomorphic(path4, relabel(path4, [2, 0, 3, 1]))
    assert check_isomorphic(star4, relabel(star4, [4, 0, 1, 2, 3]))   # disconnected
    assert check_isomorphic(bowtie, relabel(bowtie, [3, 1, 4, 0, 2]))
    assert check_isomorphic([], [])
    assert check_isomorphic([[]], [[]])

    # different degree sequences: star K1,3 vs path P4, bowtie vs C5
    star3 = [[1, 2, 3], [0], [0], [0]]
    assert not check_isomorphic(star3, path4)
    assert not check_isomorphic(bowtie, c5)

    # C6 vs two disjoint triangles: same degrees, different components (and triangles)
    pair_a, pair_b = c6, two_triangles()
    assert degrees(pair_a) == degrees(pair_b) == [2, 2, 2, 2, 2, 2]
    assert triangle_count(pair_a) == 0
    assert triangle_count(pair_b) == 2           # one triangle per component
    assert component_sizes(pair_a) == [6]
    assert component_sizes(pair_b) == [3, 3]
    assert not check_isomorphic(pair_a, pair_b)

    # K3,3 vs triangular prism: same degrees [3]*6, same components [6], triangles differ
    assert degrees(k33()) == degrees(prism()) == [3, 3, 3, 3, 3, 3]
    assert component_sizes(k33()) == component_sizes(prism()) == [6]
    assert triangle_count(k33()) == 0
    assert triangle_count(prism()) == 2
    assert not check_isomorphic(k33(), prism())

    # documented insufficiency: the cube Q3 and the Moebius ladder V8 pass all
    # three checks yet are not isomorphic (Q3 is bipartite, V8 is not)
    cube, v8 = cube_q3(), mobius_v8()
    assert check_isomorphic(cube, v8)
    assert degrees(cube) == degrees(v8) == [3] * 8
    assert triangle_count(cube) == triangle_count(v8) == 0
    assert component_sizes(cube) == component_sizes(v8) == [8]
    assert is_bipartite(cube)
    assert not is_bipartite(v8)
    assert diameter(cube) == 3
    assert diameter(v8) == 2

    # random relabelings never break the invariants; independent oracles agree
    seed = 20261016
    for graph in lcg_graphs(40):
        perm, seed = lcg_permutation(len(graph), seed)
        relabelled = relabel(graph, perm)
        assert check_isomorphic(graph, relabelled)
        assert degrees(graph) == degrees(relabelled)
        assert triangle_count(graph) == triangle_count(relabelled)
        assert component_sizes(graph) == component_sizes_union_find(graph)
        assert component_sizes(relabelled) == component_sizes_union_find(relabelled)
        inverse = [0] * len(perm)
        for old, new in enumerate(perm):
            inverse[new] = old
        assert relabel(relabelled, inverse) == graph


if __name__ == "__main__":
    main()
    print("PASS")
