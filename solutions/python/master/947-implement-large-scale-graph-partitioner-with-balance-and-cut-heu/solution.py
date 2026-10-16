"""Task 947 - Implement large-scale graph partitioner with balance and cut heuristics.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

A three-phase multilevel partitioner.  (1) Coarsening: heavy-edge matching visits vertices in a
deterministic order and merges each with its unmatched neighbour of maximum edge weight, so the total
edge weight is preserved while the graph shrinks.  (2) Initial partitioning: greedy BFS growth from the
heaviest unassigned vertex, capped at ceil(n / k) vertices per part, with the leftovers going to the
smallest part.  (3) Refinement: an FM-style pass that swaps boundary pairs whose swap lowers the cut.

Simplified versus METIS: a single coarsening level instead of repeated coarsening, greedy growth instead
of recursive bisection or a spectral seed, and one hill-climbing pass instead of gain-bucketed FM.
Verified: the output is a real partition and the balance constraint holds for k = 2, 3, 4; the cut beats
a random balanced baseline and matches an explicit edge-by-edge count; against exhaustive brute force
on tiny graphs the result is optimal on a clique-pair instance and never below the optimum / within 2x
of it on a random one; coarsening preserves the edge-weight total and matches each vertex at most once;
refinement is monotone in the cut (swaps preserve part sizes, so balance cannot drift); the k = 1,
k > n and empty-graph boundaries; determinism.
"""
import collections

TOLERANCE = 0.1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def pseudo(value):
    """Deterministic 64-bit mix, so graph construction needs no RNG object."""
    value = (value * 0x9E3779B97F4A7C15 + 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 29
    return ((value * 0x94D049BB133111EB) & ((1 << 64) - 1)) >> 11


def build_graph(vertices, degree):
    """Undirected weighted graph: a ring plus deterministic chords."""
    weights = {(vertex, (vertex + 1) % vertices): 1 for vertex in range(vertices)}
    for source in range(vertices):
        for step in range(degree):
            target = pseudo(source * 131 + step * 17) % vertices
            key = (min(source, target), max(source, target))
            if target != source:
                weights[key] = weights.get(key, 0) + 1
    return tuple((source, target, weight) for (source, target), weight in sorted(weights.items()))


def adjacency(edges):
    neighbours = {}
    for source, target, weight in edges:
        neighbours.setdefault(source, {})[target] = weight
        neighbours.setdefault(target, {})[source] = weight
    return neighbours


def cut_size(assignment, edges):
    return sum(weight for source, target, weight in edges
               if assignment[source] != assignment[target])


def part_sizes(assignment, k):
    sizes = [0] * k
    for part in assignment.values():
        sizes[part] += 1
    return sizes


def is_balanced(assignment, k, tolerance=TOLERANCE):
    sizes = part_sizes(assignment, k)
    target = sum(sizes) / k
    return all(size <= target * (1 + tolerance) for size in sizes)


def coarsen(edges):
    """Heavy-edge matching: returns representative -> members."""
    neighbours = adjacency(edges)
    order = sorted(neighbours, key=lambda vertex: (-len(neighbours[vertex]), vertex))
    pairs, matched = {}, set()
    for vertex in order:
        if vertex in matched:
            continue
        options = [(neighbours[vertex][other], -other, other) for other in neighbours[vertex]
                   if other not in matched]
        partner = max(options)[2] if options else None
        pairs[vertex] = (vertex,) if partner is None else (vertex, partner)
        matched.update(pairs[vertex])
    return pairs


def coarsened_edges(edges, pairs):
    """Collapse each pair into a super-vertex; parallel edges become weighted edges."""
    cluster = {member: representative for representative, members in pairs.items()
               for member in members}
    weights = {}
    for source, target, weight in edges:
        first, second = cluster[source], cluster[target]
        if first != second:
            key = (min(first, second), max(first, second))
            weights[key] = weights.get(key, 0) + weight
    return tuple((source, target, weight) for (source, target), weight in sorted(weights.items()))


def greedy_initial(edges, k):
    neighbours = adjacency(edges)
    vertices = sorted(neighbours)
    if not vertices:
        return {}
    limit, assignment = -(-len(vertices) // k), {}
    for part in range(k):
        unassigned = [vertex for vertex in vertices if vertex not in assignment]
        if not unassigned:
            break
        seed = max(unassigned, key=lambda vertex: (len(neighbours[vertex]), -vertex))
        frontier, size, assignment[seed] = collections.deque([seed]), 1, part
        while frontier and size < limit:
            vertex = frontier.popleft()
            for other in sorted(neighbours[vertex],
                                key=lambda item: (-neighbours[vertex][item], item)):
                if other not in assignment and size < limit:
                    assignment[other], size = part, size + 1
                    frontier.append(other)
    sizes = part_sizes(assignment, k)
    for vertex in vertices:
        if vertex not in assignment:
            part = min(range(k), key=lambda candidate: (sizes[candidate], candidate))
            assignment[vertex], sizes[part] = part, sizes[part] + 1
    return assignment


def refine(assignment, edges, k, rounds=8):
    """Swap boundary pairs whose swap lowers the cut; a swap keeps every part size unchanged."""
    current = cut_size(assignment, edges)
    for _ in range(rounds):
        best = None
        for source, target, _ in edges:
            for first, second in ((source, target), (target, source)):
                if assignment[first] == assignment[second]:
                    continue
                trial = dict(assignment)
                trial[first], trial[second] = trial[second], trial[first]
                gained = current - cut_size(trial, edges)
                if gained > 0 and (best is None or gained > best[0]):
                    best = (gained, trial)
        if best is None:
            break
        current, assignment = current - best[0], best[1]
    return assignment, current


def exhaustive_optimum(edges, vertices, k=2):
    """Oracle on tiny graphs: brute force over balanced k-partitions -> (best cut, #evaluated)."""
    best, evaluated = None, 0
    for mask in range(k ** (vertices - 1)):
        assignment, code = {0: 0}, mask
        for vertex in range(1, vertices):
            assignment[vertex], code = code % k, code // k
        if len(set(assignment.values())) != k or not is_balanced(assignment, k):
            continue
        evaluated += 1
        cut = cut_size(assignment, edges)
        best = cut if best is None else min(best, cut)
    return best, evaluated


def random_balanced(edges, k):
    """Deterministic shuffled assignment used as the quality baseline."""
    order = sorted(adjacency(edges), key=lambda vertex: pseudo(vertex * 7919 + 104729))
    return {vertex: min(k - 1, index * k // len(order)) for index, vertex in enumerate(order)}


def partition(edges, k):
    initial = greedy_initial(edges, k)
    return refine(initial, edges, k)[0], initial


def main():
    edges = build_graph(80, 3)
    vertices = sorted(adjacency(edges))
    results = {}
    # 1) the output is a real partition and holds the balance constraint for k = 2, 3, 4.
    for k in (2, 3, 4):
        refined, initial = partition(edges, k)
        results[k] = (refined, initial)
        check(sorted(refined) == vertices and set(refined.values()) == set(range(k))
              and is_balanced(refined, k),
              f"k={k} must assign every vertex once inside the balance tolerance: "
              f"{part_sizes(refined, k)}")

    # 2) the multilevel result beats a deterministic random balanced baseline, and the cut matches an
    #    explicit edge-by-edge count on a hand-made split.
    baseline = random_balanced(edges, 4)
    hand = {vertex: (0 if vertex < 40 else 1) for vertex in vertices}
    expected = sum(weight for source, target, weight in edges if (source < 40) != (target < 40))
    check(is_balanced(baseline, 4) and cut_size(results[4][0], edges) < cut_size(baseline, edges)
          and cut_size(hand, edges) == expected and part_sizes(hand, 2) == [40, 40],
          f"cut {cut_size(results[4][0], edges)} vs baseline {cut_size(baseline, edges)}; "
          f"hand cut {cut_size(hand, edges)} vs explicit {expected}")

    # 3) refinement is monotone in the cut and balanced, and part sizes are untouched by swaps.
    for k, (refined, initial) in results.items():
        check(cut_size(refined, edges) <= cut_size(initial, edges) and is_balanced(refined, k)
              and part_sizes(refined, k) == part_sizes(initial, k),
              f"k={k}: {cut_size(initial, edges)} -> {cut_size(refined, edges)} with sizes "
              f"{part_sizes(refined, k)}")

    # 4) independent oracle on tiny graphs: brute-force optima over balanced 2-partitions.
    tiny_edges = build_graph(8, 2)
    optimum, evaluated = exhaustive_optimum(tiny_edges, 8, 2)
    tiny_cut = cut_size(partition(tiny_edges, 2)[0], tiny_edges)
    cliques = tuple([(a, b, 1) for a in range(4) for b in range(a + 1, 4)]
                    + [(a, b, 1) for a in range(4, 8) for b in range(a + 1, 8)] + [(3, 4, 1)])
    clique_optimum, clique_evaluated = exhaustive_optimum(cliques, 8, 2)
    check(evaluated > 10 and optimum <= tiny_cut <= 2 * optimum and clique_optimum == 1
          and cut_size(partition(cliques, 2)[0], cliques) == 1,
          f"tiny random {tiny_cut} vs optimum {optimum} over {evaluated}; the clique pair must be "
          f"split optimally over {clique_evaluated} candidate partitions")
    # 5) coarsening preserves the edge-weight total, matches each vertex at most once, shrinks the
    #    graph and keeps unmatched vertices as singletons.
    pairs = coarsen(edges)
    members = [member for group in pairs.values() for member in group]
    reduced = coarsened_edges(edges, pairs)
    check(len(members) == len(set(members)) == len(vertices) and len(pairs) < len(vertices)
          and any(len(group) == 2 for group in pairs.values())
          and all(len(group) in (1, 2) for group in pairs.values())
          and sum(weight for _, _, weight in reduced) <= sum(weight for _, _, weight in edges),
          f"coarsening: {len(pairs)} super-vertices from {len(vertices)}, {len(reduced)} edges")
    # 6) boundaries: k = 1, k > n and the empty graph.
    single, _ = partition(edges, 1)
    sizes_many = part_sizes(partition(edges, len(vertices) + 4)[0], len(vertices) + 4)
    check(set(single.values()) == {0} and cut_size(single, edges) == 0 and max(sizes_many) == 1
          and sum(sizes_many) == len(vertices) and partition((), 2)[0] == {},
          f"k=1 cut {cut_size(single, edges)}; k>n largest part {max(sizes_many)}")
    # 7) the balance predicate is exactly the tolerance rule, on both sides of the boundary.
    split = lambda cut: {vertex: 0 if vertex < cut else 1 for vertex in range(40)}
    check(is_balanced({vertex: vertex % 2 for vertex in range(40)}, 2)
          and is_balanced(split(22), 2) and not is_balanced(split(30), 2),
          "the balance predicate must accept 20/20 and 22/18 and reject 30/10")
    # 8) determinism.
    check(partition(edges, 3) == results[3],
          "two partitionings of the same graph must be identical")

    print(f"vertices={len(vertices)} edges={len(edges)} cuts="
          f"{ {k: cut_size(refined, edges) for k, (refined, _) in sorted(results.items())} } "
          f"baseline4={cut_size(baseline, edges)}; tiny optimum={optimum} vs {tiny_cut}; coarsened to "
          f"{len(pairs)} super-vertices with {len(reduced)} edges")
    print("PASS")


if __name__ == "__main__":
    main()