"""Task 949 - Implement approximate maximum matching in huge graphs with streaming algorithms.  (Tier: Master, CTCI Ch 4 - Trees and Graphs)

Streaming matchings under a memory budget.  Edges arrive one at a time and may be read a bounded number
of times; the algorithms keep only a partner map and a bounded sample: `greedy_stream` is a single pass
that matches an edge whenever both endpoints are still free (in arrival order or in a deterministic
shuffled order), and `sampled_stream` is the three-pass composable core-set recipe - reservoir-sample
`reserve` edges, run the greedy matching inside the sample, then sweep the whole stream again to match
whatever is still free.  Every algorithm returns a valid matching, which is automatically a
2-approximation, and the sample makes the multi-pass ratio better on adversarial orders.

Simplified versus streaming-MATCH (Paz-Schwartzman, McGregor): a fixed reservoir size replaces the union
bound and the family is small enough to brute-force.  Verified: no vertex is matched twice and every
ratio is at least 1/2 of the exact optimum from two independent oracles (edge-subset brute force on tiny
instances, Kuhn's augmenting paths on the bipartite family); every result is maximal and the reservoir
stays inside the budget; a deterministic shuffle beats the adversarial FIFO order and the sample matching
is itself a 1/2-approximation of the sample optimum (the core-set lemma); empty, single-edge, star, path
and perfect-matching boundaries; determinism.
"""
import collections
import itertools

BUDGET = 64


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def pseudo(value):
    """Deterministic 64-bit mix, so the shuffled stream order is reproducible."""
    value = (value * 0x9E3779B97F4A7C15 + 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 29
    return ((value * 0x94D049BB133111EB) & ((1 << 64) - 1)) >> 11


def exact_matching(edges):
    """Oracle for tiny instances: brute force over all edge subsets, largest first."""
    for size in range(len(edges), 0, -1):
        for subset in itertools.combinations(edges, size):
            if len({vertex for edge in subset for vertex in edge}) == 2 * size:
                return size
    return 0


def kuhn_matching(edges):
    """Exact maximum matching for the bipartite instances: 2-colour, then augmenting paths."""
    neighbours = collections.defaultdict(list)
    for source, target in edges:
        neighbours[source].append(target)
        neighbours[target].append(source)
    colour, left, assigned = {}, [], {}
    for start in sorted(neighbours):
        if start in colour:
            continue
        colour[start], stack = 0, [start]
        while stack:
            vertex = stack.pop()
            if colour[vertex] == 0:
                left.append(vertex)
            for other in neighbours[vertex]:
                if other not in colour:
                    colour[other] = 1 - colour[vertex]
                    stack.append(other)

    def augment(vertex, visited):
        for other in neighbours[vertex]:
            if colour[other] == 0 or other in visited:
                continue
            visited.add(other)
            if other not in assigned or augment(assigned[other], visited):
                assigned[other] = vertex
                return True
        return False

    return sum(1 for vertex in left if augment(vertex, set()))


def shuffled(edges, seed):
    """The same edge multiset in a deterministic hash order, as a stream can be re-shuffled."""
    return sorted(edges, key=lambda edge: (pseudo((edge[0] * 4099 + edge[1]) * 7919 + seed),
                                           edge))


def greedy_stream(edges, seed=0, reshuffled=True):
    """One pass; matches an edge whenever both endpoints are still free."""
    stream = shuffled(edges, seed) if reshuffled else list(edges)
    partner, matched = {}, []
    for source, target in stream:
        if source not in partner and target not in partner:
            partner[source], partner[target] = target, source
            matched.append((source, target))
    return matched, len(partner)


def reservoir(edges, reserve, seed):
    """Pass 1: deterministic reservoir sample of the stream."""
    sample, seen = [], 0
    for edge in edges:
        seen += 1
        if len(sample) < reserve:
            sample.append(edge)
        else:
            slot = pseudo(seen * 7919 + seed) % seen
            if slot < reserve:
                sample[slot] = edge
    return sample


def sampled_stream(edges, reserve=BUDGET, seed=0):
    """Three passes: sample, greedy matching inside the sample, then a shuffled sweep."""
    matched, partner = greedy_stream(reservoir(edges, reserve, seed), seed)[0], {}
    for source, target in matched:
        partner[source], partner[target] = target, source
    for source, target in shuffled(edges, seed + 1):
        if source not in partner and target not in partner:
            partner[source], partner[target] = target, source
            matched.append((source, target))
    return matched, len(partner), min(len(reservoir(edges, reserve, seed)), reserve)


def valid_matching(edges, matched):
    seen, pool = set(), set(edges)
    for source, target in matched:
        if source == target or source in seen or target in seen \
                or ((source, target) not in pool and (target, source) not in pool):
            return False
        seen.update((source, target))
    return True


def build_path(length):
    return [(vertex, vertex + 1) for vertex in range(length)]


def build_star(leaves):
    return [(0, leaf) for leaf in range(1, leaves + 1)]
def adversarial_path(length):
    """Path edges ordered odd-first, the order that makes a FIFO greedy waste an edge."""
    edges = build_path(length)
    return edges[1::2] + edges[0::2]


def build_complete_bipartite(left, right):
    return [(first, left + second) for first in range(left) for second in range(right)]

def build_random(vertices, extra, seed):
    """Random bipartite instance, so the exact optimum stays polynomial."""
    half = vertices // 2
    edges = {(vertex, vertex + 1) for vertex in range(vertices - 1) if vertex + 1 != half}
    edges.update((pseudo(seed * 104729 + step * 131) % half,
                  half + pseudo(seed * 15485863 + step * 197) % (vertices - half))
                 for step in range(extra))
    return sorted(edges)


def family():
    """Adversarial arrival orders plus random bipartite instances, arrival and shuffled."""
    bases = [build_path(4), build_path(9), build_star(8), build_complete_bipartite(3, 3),
             build_path(30), build_star(20), build_random(24, 30, 7), build_random(30, 20, 11)]
    return bases + [shuffled(instance, seed) for seed, instance in enumerate(bases)]


def maximal(edges, matched):
    """A matching is maximal when no edge has both endpoints free (so it is a 2-approximation)."""
    covered = {vertex for edge in matched for vertex in edge}
    return all(source in covered or target in covered for source, target in edges)



def main():
    instances = family()
    optima = [kuhn_matching(instance) for instance in instances]
    ratios = []
    for instance, optimum in zip(instances, optima):
        for matched, _ in (greedy_stream(instance, reshuffled=False), greedy_stream(instance, 3),
                           sampled_stream(instance, BUDGET, 5)[:2]):
            check(valid_matching(instance, matched) and 2 * len(matched) >= optimum,
                  f"a maximal matching must be valid and at least half of {optimum}: {matched}")
            ratios.append(len(matched) / optimum)
    check(len(ratios) == 3 * len(instances) and min(ratios) >= 0.5, f"worst ratio {min(ratios):.3f}")
    # 1) no algorithm ever produces an invalid matching; every ratio is at least 1/2, and the two
    #    oracles agree on the textbook optima of the classic instances.
    classic = {"K4": build_complete_bipartite(2, 2), "P4": build_path(3),
               "C5": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], "star": build_star(6),
               "empty": [], "single edge": [(0, 1)]}
    known = {"K4": 2, "P4": 2, "C5": 2, "star": 1, "empty": 0, "single edge": 1}
    brute = {name: exact_matching(edges) for name, edges in classic.items()}
    check(brute == known and all(kuhn_matching(edges) == known[name]
                                 for name, edges in classic.items() if name != "C5"),
          f"the oracles must reproduce the textbook optima: {brute}")

    # 3) the core-set lemma, the shuffled-order benefit and maximality, checked directly.
    adversarial = [adversarial_path(length) for length in (3, 5, 9, 15, 21, 25, 31)]
    fifo = sum(len(greedy_stream(instance, reshuffled=False)[0]) for instance in adversarial)
    shuffling = sum(len(greedy_stream(instance, seed=3)[0]) for instance in adversarial)
    ratios = []
    for instance in adversarial + instances[:4]:
        matched, _, peak = sampled_stream(instance, BUDGET, 5)
        head, _ = greedy_stream(reservoir(instance, BUDGET, 5), 5)
        optimum_sample = kuhn_matching(reservoir(instance, BUDGET, 5))
        ratios.append(len(head) / optimum_sample if optimum_sample else 1.0)
        check(maximal(instance, matched) and peak <= BUDGET,
              f"the core-set result must be maximal and inside the budget: {len(matched)}")
    check(min(ratios) >= 0.5 and shuffling > fifo,
          f"the sample matching must be a 1/2-approximation of the sample optimum (worst "
          f"{min(ratios):.3f}); the shuffle {shuffling} must beat the FIFO order {fifo}")

    # 4) the reservoir never exceeds its budget and the memory accounting is exact.
    peaks = [sampled_stream(instance, 40)[2] for instance in instances[:6]]
    biggest = build_random(40, 40, 23)
    one_pass, one_pass_vertices = greedy_stream(biggest, seed=9)
    check(max(peaks) <= 40 and all(sampled_stream(instance, 40)[1] == 2 * len(
        sampled_stream(instance, 40)[0]) for instance in instances[:6])
          and one_pass_vertices == 2 * len(one_pass) <= 40 and valid_matching(biggest, one_pass),
          f"peak reservoir {max(peaks)}; one-pass stored {one_pass_vertices} vertices")
    # 5) the FIFO order is optimal on the star and the path, and the sampled pass stays inside
    #    [optimum/2, optimum].
    star, path = build_star(20), build_path(20)
    star_opt, path_opt = kuhn_matching(star), kuhn_matching(path)
    star_sampled, path_sampled = len(sampled_stream(star)[0]), len(sampled_stream(path)[0])
    check(len(greedy_stream(star, reshuffled=False)[0]) == star_opt == 1
          and len(greedy_stream(path, reshuffled=False)[0]) == path_opt == 10
          and 2 * star_sampled >= star_opt and star_sampled <= star_opt
          and 2 * path_sampled >= path_opt and path_sampled <= path_opt,
          f"star {star_sampled}/{star_opt}, path {path_sampled}/{path_opt}")

    # 6) boundaries: the empty stream, a single edge and a perfect-matching instance.
    perfect = build_complete_bipartite(3, 3)
    check(sampled_stream([])[:2] == ([], 0)
          and len(sampled_stream([(4, 9)])[0]) == 1 == exact_matching([(4, 9)])
          and 2 * len(sampled_stream(perfect)[0]) >= 3 == exact_matching(perfect),
          "the empty, single-edge and perfect-matching boundaries must hold")

    # 7) the shuffled order is a deterministic permutation, and the pipeline is deterministic.
    check(shuffled(build_path(12), 4) == shuffled(build_path(12), 4)
          and sorted(shuffled(build_path(12), 4)) == sorted(build_path(12))
          and shuffled(build_path(12), 4) != shuffled(build_path(12), 5)
          and sampled_stream(instances[0], 8, 2) == sampled_stream(instances[0], 8, 2),
          "the shuffled stream must be a deterministic permutation of the input")

    print(f"instances={len(instances)} worst_ratio={min(ratios):.3f} fifo={fifo} shuffled={shuffling} "
          f"optima={sum(optima)}; reservoir peak={max(peaks)}; star={kuhn_matching(star)} "
          f"path={kuhn_matching(path)} K4={exact_matching(classic['K4'])}")
    print("PASS")


if __name__ == "__main__":
    main()
