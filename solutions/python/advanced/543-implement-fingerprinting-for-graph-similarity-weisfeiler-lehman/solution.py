"""Task 543 - Implement fingerprinting for graph similarity (Weisfeiler-Lehman sketch). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

The one-dimensional Weisfeiler-Lehman colour refinement: every vertex starts with its degree as
colour, and each round replaces the colour by a canonical renumbering of the pair (own colour,
sorted multiset of neighbour colours). The renumbering sorts the signatures, so colours depend only
on the isomorphism class of the labelled structure and never on vertex names or adjacency order.
Two fingerprints are produced: a per-graph sketch (canonical colour vectors, their histograms and a
stable polynomial hash over the rounds) and the standard joint refinement, which runs the test on
the *disjoint union* of two graphs and compares the per-round colour histograms - if any round
differs, the graphs are provably non-isomorphic.

The declared limitation is exercised, not hidden: C6 and two disjoint triangles are 2-regular and
stay indistinguishable under 1-WL, so both are reported as equivalent while the brute-force
bijection search says otherwise; the file counts such false positives on random instances and,
conversely, asserts the one-directional guarantee that a difference at any round is never a false
negative (checked against brute-force isomorphism on every sampled pair).

Checks: the sketch of a graph is invariant under relabelling and under shuffling the adjacency
lists, and equals the sketch of an isomorphic copy; a stable-hash equality test for the sampled
pairs; the hand-built pair C6 versus 2*C3 that 1-WL cannot separate; stars/paths/triangles
separated by the first round alone; the colour classes refine monotonically (the number of classes
never decreases, their sizes sum to n, and the partition is stable after n rounds - the documented
bound); the WL colour classes agree with an independently written partition-refinement loop whose
cells are verified to have identical neighbour-cell profiles; and the round histograms sum to n.
"""
import collections
import itertools


def canonicalize(values):
    """Renumber hashable values to 0..k-1 by sorted order; deterministic and label-independent."""
    unique = sorted(set(values))
    mapping = {value: index for (index, value) in enumerate(unique)}
    return [mapping[value] for value in values], len(unique)


class Graph:
    def __init__(self, n, edges):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.edges = []
        for (u, v) in edges:
            self.add_edge(u, v)

    def add_edge(self, u, v):
        assert 0 <= u < self.n and 0 <= v < self.n
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.edges.append((min(u, v), max(u, v)))

def wl_rounds(graph, iterations):
    """Colour vectors after 0..iterations refinement rounds (round 0 = degrees)."""
    colours, classes = canonicalize([len(graph.adj[v]) for v in range(graph.n)])
    history = [colours]
    class_counts = [classes]
    for _ in range(iterations):
        signatures = [(colours[v], tuple(sorted(colours[w] for w in graph.adj[v])))
                      for v in range(graph.n)]
        colours, classes = canonicalize(signatures)
        history.append(colours)
        class_counts.append(classes)
    return history, class_counts


def wl_sketch(graph, iterations):
    """Per-graph fingerprint: colour histograms per round plus a stable hash of the rounds."""
    history, class_counts = wl_rounds(graph, iterations)
    histograms = [tuple(sorted(collections.Counter(colours).items())) for colours in history]
    hashed = 0
    for round_index, histogram in enumerate(histograms):
        hashed = (hashed * 1000003 + round_index * 7919 + len(histogram) * 104729) % (2 ** 61 - 1)
        for (colour, count) in histogram:
            hashed = (hashed * 1000003 + colour * 6151 + count) % (2 ** 61 - 1)
    return {"history": history, "histograms": histograms, "classes": class_counts,
            "hash": hashed}


def wl_equivalent(graph1, graph2, iterations):
    """Joint refinement on the disjoint union: True iff no round separates the two graphs."""
    n1 = graph1.n
    n2 = graph2.n
    total = n1 + n2
    adj = [[] for _ in range(total)]
    for v in range(n1):
        adj[v] = list(graph1.adj[v])
    for v in range(n2):
        adj[n1 + v] = [n1 + w for w in graph2.adj[v]]
    colours, _ = canonicalize([len(adj[v]) for v in range(total)])
    for _ in range(iterations):
        signatures = [(colours[v], tuple(sorted(colours[w] for w in adj[v])))
                      for v in range(total)]
        colours, _ = canonicalize(signatures)
        left = collections.Counter(colours[:n1])
        right = collections.Counter(colours[n1:])
        if left != right:
            return False
    return True


def partition_refinement(graph):
    """Independent refinement loop over an explicit vertex partition."""
    n = graph.n
    partition = [0] * n                      # start: everything in one cell
    cells = [frozenset(range(n))]
    while True:
        signature = [(partition[v], tuple(sorted(partition[w] for w in graph.adj[v])))
                     for v in range(n)]
        new_partition, _ = canonicalize(signature)
        new_cells = []
        for colour in sorted(set(new_partition)):
            new_cells.append(frozenset(v for v in range(n) if new_partition[v] == colour))
        if set(new_cells) == set(cells):
            return frozenset(cells)
        partition = new_partition
        cells = new_cells


def classes_from_colours(colours, n):
    return frozenset(frozenset(v for v in range(n) if colours[v] == colour)
                     for colour in set(colours))


def isomorphic_brute(g1, g2):
    """Exhaustive isomorphism for tiny graphs, by trying every bijection."""
    if g1.n != g2.n or len(g1.edges) != len(g2.edges):
        return False
    n = g1.n
    target_edges = {(min(u, v), max(u, v)) for (u, v) in g2.edges}
    for permutation in itertools.permutations(range(n)):
        images = set()
        for (u, v) in g1.edges:
            a, b = permutation[u], permutation[v]
            images.add((min(a, b), max(a, b)))
        if images == target_edges:
            return True
    return False


def relabel(graph, permutation):
    n = graph.n
    return Graph(n, [(permutation[u], permutation[v]) for (u, v) in graph.edges])


def lcg(seed):
    return (seed * 75 + 74) % 65537


def cycle(n):
    return Graph(n, [(v, (v + 1) % n) for v in range(n)])


def complete(n):
    return Graph(n, [(i, j) for i in range(n) for j in range(i + 1, n)])


def star(n):
    return Graph(n, [(0, v) for v in range(1, n)])


def lcg_graph(n, edge_probability, seed):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            seed = lcg(seed)
            if seed % 100 < edge_probability:
                edges.append((i, j))
    return Graph(n, edges), seed


def lcg_permutation(n, seed):
    permutation = list(range(n))
    for i in range(n - 1, 0, -1):
        seed = lcg(seed)
        j = seed % (i + 1)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation, seed


def main():
    # --- Invariance of the sketch -------------------------------------------------------
    seed = 2024
    for n in (4, 6, 8, 10):
        for probability in (30, 60):
            graph, seed = lcg_graph(n, probability, seed)
            reference = wl_sketch(graph, n)
            assert wl_sketch(graph, n)["hash"] == reference["hash"], "deterministic"
            permutation, seed = lcg_permutation(n, seed)
            moved = relabel(graph, permutation)
            assert wl_sketch(moved, n)["hash"] == reference["hash"], "relabelling"
            # The colour vectors are indexed by vertex, so a relabelling permutes them; the
            # histograms and the class counts are the label-independent part of the sketch.
            assert wl_sketch(moved, n)["histograms"] == reference["histograms"]
            assert wl_sketch(moved, n)["classes"] == reference["classes"]
            assert sorted(wl_sketch(moved, n)["history"][-1]) == \
                sorted(reference["history"][-1])
            shuffled = Graph(n, list(reversed(graph.edges)))
            assert wl_sketch(shuffled, n)["hash"] == reference["hash"], "edge order"
            assert wl_equivalent(graph, moved, n), "isomorphic copies never separate"
            assert isomorphic_brute(graph, moved)

    # --- Hand-checked separations and the documented blindness --------------------------
    c6 = cycle(6)
    two_triangles = Graph(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert not isomorphic_brute(c6, two_triangles), "brute force separates them"
    assert wl_equivalent(c6, two_triangles, 6), \
        "1-WL cannot separate two 2-regular graphs - the classic limitation"
    assert wl_sketch(c6, 6)["classes"] == wl_sketch(two_triangles, 6)["classes"] == \
        [1] * 7, "both stay a single colour class in every round"
    assert wl_sketch(c6, 6)["hash"] != wl_sketch(star(6), 6)["hash"], "degrees differ at round 0"
    path6 = Graph(6, [(v, v + 1) for v in range(5)])
    assert wl_sketch(path6, 6)["histograms"][0] != wl_sketch(star(6), 6)["histograms"][0], \
        "the degree histogram alone separates a path from a star"
    assert not wl_equivalent(path6, star(6), 6)
    assert not wl_equivalent(cycle(5), star(5), 5)
    assert not wl_equivalent(complete(4), cycle(4), 4)
    assert not wl_equivalent(star(5), cycle(4), 4), "different sizes"
    assert wl_equivalent(c6, cycle(6), 6) and isomorphic_brute(c6, cycle(6))

    # Round counting: a 2-regular graph needs no refinement, a path keeps splitting.
    path5 = Graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    sketch = wl_sketch(path5, 5)
    assert sketch["classes"][0] == 2, "degrees 1 and 2"
    assert sketch["classes"][1] == 3, "the two ends split from the inner vertices"
    counts = sketch["classes"]
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)), \
        "refinement only ever splits classes"
    assert counts[-1] == counts[-2] == 3, \
        "P5 stabilises at its three-class equitable partition {ends}, {inner pair}, {centre}"
    assert sketch["classes"][0] == 2, "the path starts with the degree partition"
    for (colour, count) in sketch["histograms"][3]:
        assert count >= 1
    for histogram in sketch["histograms"]:
        assert sum(count for (_, count) in histogram) == 5, "histograms sum to n"

    # Stability within n rounds for every sampled graph.
    seed = 555
    for n in (5, 7, 9, 11):
        graph, seed = lcg_graph(n, 45, seed)
        history, class_counts = wl_rounds(graph, n)
        assert history[-1] == history[-2], "the colouring is stable after n rounds"
        assert class_counts[-1] == class_counts[-2]
        assert len(set(history[-1])) == class_counts[-1]
        for round_index, colours in enumerate(history[:-1]):
            finer = classes_from_colours(colours, n)
            coarser = classes_from_colours(history[round_index + 1], n)
            for cell in coarser:
                assert any(cell <= other for other in finer), \
                    "each new cell is contained in an old one"

    # --- The colour classes equal an independently written partition refinement ---------
    seed = 31337
    for n in (4, 6, 7, 9):
        graph, seed = lcg_graph(n, 50, seed)
        history, _ = wl_rounds(graph, n)
        wl_partition = classes_from_colours(history[-1], n)
        refined = partition_refinement(graph)
        assert wl_partition == refined, "the two refinement implementations agree"
        # Stability of the independent partition: cell membership determines the neighbour
        # profile exactly.
        index_of = {}
        for (index, cell) in enumerate(sorted(refined, key=lambda c: sorted(c))):
            for v in cell:
                index_of[v] = index
        profiles = {}
        for v in range(n):
            profiles[v] = (index_of[v], tuple(sorted(index_of[w] for w in graph.adj[v])))
        for cell in refined:
            values = {profiles[v] for v in cell}
            assert len(values) == 1, "vertices in a stable cell have identical profiles"

    # --- Soundness of the WL verdict against brute force -------------------------------
    seed = 6060
    graphs = [c6, two_triangles]
    for n in (6, 6, 6, 6, 5, 5, 4, 4):
        graph, seed = lcg_graph(n, 55, seed)
        graphs.append(graph)
    for source in (2, 3, 4, 5):                 # isomorphic copies, as positive controls
        permutation, seed = lcg_permutation(graphs[source].n, seed)
        graphs.append(relabel(graphs[source], permutation))
    false_positives = 0
    separations = 0
    positives = 0
    for i in range(len(graphs)):
        for j in range(len(graphs)):
            g1, g2 = graphs[i], graphs[j]
            equivalent = wl_equivalent(g1, g2, 6)
            actually = isomorphic_brute(g1, g2)
            if equivalent:
                positives += 1
                if not actually:
                    false_positives += 1
                else:
                    assert wl_sketch(g1, 6)["hash"] == wl_sketch(g2, 6)["hash"], \
                        "isomorphic graphs share the fingerprint hash"
            else:
                separations += 1
                assert not actually, "a difference at some round rules out isomorphism"
                assert wl_sketch(g1, 6)["hash"] != wl_sketch(g2, 6)["hash"], \
                    "this hash distinguishes every pair the refinement separates"
    assert separations > 40, "the sampler produced plenty of separated pairs"
    assert positives >= 12, "including the four deliberately isomorphic copies"
    assert false_positives >= 1, \
        "C6 versus two triangles is a documented 1-WL blind spot, and it is in the sample"


if __name__ == "__main__":
    main()
    print("PASS")
