"""Task 537 - Implement approximate TSP heuristics (2-opt / 3-opt). (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)

Local search for the symmetric Euclidean travelling salesman: a tour is a cyclic permutation of
the cities and its length is the sum of the Euclidean distances along the cycle. Three
constructive/improvement stages are implemented. ``nn_tour`` is the nearest-neighbour
construction with deterministic tie-breaking by city index. ``two_opt_tour`` repeatedly replaces
two edges (a, b) and (c, d) by (a, c) and (b, d), which is exactly the reversal of one contiguous
segment. ``three_opt_tour`` performs best-improvement over every triple of removed edges: for
positions i < j < k the tour is split into four consecutive blocks A B C D and the seven
reconnections (B, C), (rev B, C), (B, rev C), (rev B, rev C), (C, B), (rev C, B), (C, rev B) are
enumerated, each candidate checked to be a permutation before its length is compared. Because the
(B, rev C) and (rev B, C) cases reproduce 2-opt exactly, a 3-opt local optimum is automatically
2-opt stable - the self-test verifies that by running 2-opt on the 3-opt output and asserting the
length does not change.

Checks: the cost function is validated against closed forms (n = 1 gives 0, n = 2 gives twice the
distance, n = 3 gives the perimeter) and against rotation/reversal invariance; the unit square,
where every heuristic must reach the brute-force optimum 4; brute-force optimal tours by
enumerating all 5040 permutations for n = 8, asserting 3-opt <= 2-opt <= nearest neighbour and
every heuristic >= the optimum; idempotence of 2-opt (no further improvement possible) and
2-opt stability of the 3-opt tour; collinear cities, where the optimum is the closed form
2 * (max - min) and 2-opt from nearest neighbour must reach it; a lower bound that every tour must
respect (any tour contains a spanning tree, so its length is at least the Kruskal MST weight);
determinism of repeated runs; and a larger 40-city instance where only the heuristics run.
"""
import itertools
import math


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tour_cost(points, tour):
    n = len(tour)
    return sum(dist(points[tour[i]], points[tour[(i + 1) % n]]) for i in range(n))


def is_tour(tour, n):
    return len(tour) == n and sorted(tour) == list(range(n))


def lcg_points(n, seed, limit=30):
    points = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        x = seed % limit
        seed = (seed * 75 + 74) % 65537
        y = seed % limit
        points.append((float(x), float(y)))
    return points


def nn_tour(points, start=0):
    """Nearest neighbour, ties broken by the smaller city index."""
    n = len(points)
    if n == 0:
        return []
    unvisited = set(range(n))
    unvisited.discard(start)
    tour = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda v: (dist(points[current], points[v]), v))
        unvisited.discard(nxt)
        tour.append(nxt)
        current = nxt
    return tour


def two_opt_tour(points, tour):
    """First-improvement 2-opt; every accepted move strictly shortens the tour."""
    best = list(tour)
    n = len(best)
    if n < 4:
        return best
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue                     # would reverse the whole cycle
                a, b = best[i], best[i + 1]
                c, d = best[j], best[(j + 1) % n]
                delta = (dist(points[a], points[c]) + dist(points[b], points[d])
                         - dist(points[a], points[b]) - dist(points[c], points[d]))
                if delta < -1e-12:
                    best[i + 1:j + 1] = reversed(best[i + 1:j + 1])
                    improved = True
    return best


def three_opt_tour(points, tour):
    """Best-improvement 3-opt using the seven segment reconnections."""
    best = list(tour)
    n = len(best)
    if n < 4:
        return best
    while True:
        best_cost = tour_cost(points, best)
        candidate = None
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    a = best[:i + 1]
                    b = best[i + 1:j + 1]
                    c = best[j + 1:k + 1]
                    d = best[k + 1:]
                    for (x, y) in ((b, c), (b[::-1], c), (b, c[::-1]), (b[::-1], c[::-1]),
                                   (c, b), (c[::-1], b), (c, b[::-1])):
                        trial = a + x + y + d
                        if trial == best:
                            continue
                        assert is_tour(trial, n), "3-opt candidate must be a permutation"
                        cost = tour_cost(points, trial)
                        if cost < best_cost - 1e-12:
                            best_cost = cost
                            candidate = trial
        if candidate is None:
            return best
        best = candidate


def brute_force_optimal(points):
    """Exhaustive optimum for small n; returns (cost, tour)."""
    n = len(points)
    if n == 0:
        return 0.0, []
    if n == 1:
        return 0.0, [0]
    if n == 2:
        return 2.0 * dist(points[0], points[1]), [0, 1]
    best_cost = None
    best_tour = None
    for perm in itertools.permutations(range(1, n)):
        tour = (0,) + perm
        cost = tour_cost(points, tour)
        if best_cost is None or cost < best_cost - 1e-12:
            best_cost = cost
            best_tour = list(tour)
    return best_cost, best_tour


def mst_weight(points):
    """Kruskal minimum spanning tree weight - a lower bound on any tour."""
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist(points[i], points[j]), i, j))
    edges.sort()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0.0
    used = 0
    for (w, i, j) in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            total += w
            used += 1
            if used == n - 1:
                break
    return total


def main():
    # --- Cost function closed forms and invariances -------------------------------------
    single = [(3.0, 4.0)]
    assert tour_cost(single, [0]) == 0.0, "a one-city tour has zero length"
    pair = [(0.0, 0.0), (3.0, 4.0)]
    assert abs(tour_cost(pair, [0, 1]) - 10.0) <= 1e-12, "there and back = 2 * 5"
    triangle = [(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)]
    assert abs(tour_cost(triangle, [0, 1, 2]) - 12.0) <= 1e-12, "3 + 4 + 5 = 12"
    assert is_tour(nn_tour(pair), 2) and nn_tour(single) == [0]

    # --- Unit square: every heuristic must reach the optimum ----------------------------
    square = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    assert abs(brute_force_optimal(square)[0] - 4.0) <= 1e-12
    nn = nn_tour(square)
    two = two_opt_tour(square, nn)
    three = three_opt_tour(square, two)
    assert abs(tour_cost(square, nn) - 4.0) <= 1e-12, "nearest neighbour is already optimal"
    assert abs(tour_cost(square, two) - 4.0) <= 1e-12
    assert abs(tour_cost(square, three) - 4.0) <= 1e-12
    assert tour_cost(square, two_opt_tour(square, list(reversed(nn)))) <= 4.0 + 1e-12

    # --- Rotation and reversal invariance of the length ---------------------------------
    tour = two_opt_tour(square, nn)
    base = tour_cost(square, tour)
    assert abs(tour_cost(square, tour[1:] + tour[:1]) - base) <= 1e-12, "rotating changes nothing"
    assert abs(tour_cost(square, list(reversed(tour))) - base) <= 1e-12, "reversing either"

    # --- Random instances against brute force ------------------------------------------
    seed = 4242
    ratios_two = []
    ratios_three = []
    for trial in range(10):
        n = 5 + trial % 4                          # 5..8
        points = lcg_points(n, seed)
        seed = (seed * 75 + 74) % 65537
        optimum, best_tour = brute_force_optimal(points)
        nn = nn_tour(points)
        two = two_opt_tour(points, nn)
        three = three_opt_tour(points, two)
        assert is_tour(nn, n) and is_tour(two, n) and is_tour(three, n)
        cost_nn = tour_cost(points, nn)
        cost_two = tour_cost(points, two)
        cost_three = tour_cost(points, three)
        assert cost_nn >= optimum - 1e-9, "no heuristic may beat the optimum"
        assert cost_two >= optimum - 1e-9 and cost_three >= optimum - 1e-9
        assert cost_two <= cost_nn + 1e-9, "2-opt never lengthens its input"
        assert cost_three <= cost_two + 1e-9, "3-opt generalises 2-opt"
        assert abs(tour_cost(points, best_tour) - optimum) <= 1e-9

        # 2-opt and 3-opt outputs are local optima: re-running cannot improve them.
        assert abs(tour_cost(points, two_opt_tour(points, two)) - cost_two) <= 1e-9, \
            "the 2-opt tour is 2-opt stable"
        assert abs(tour_cost(points, two_opt_tour(points, three)) - cost_three) <= 1e-9, \
            "the 3-opt tour is 2-opt stable"
        assert abs(tour_cost(points, three_opt_tour(points, three)) - cost_three) <= 1e-9, \
            "the 3-opt tour is 3-opt stable"

        # Every tour is at least as long as a spanning tree.
        lower = mst_weight(points)
        for value in (cost_nn, cost_two, cost_three):
            assert value >= lower - 1e-9, "a tour contains a spanning tree"
        ratios_two.append(cost_two / optimum)
        ratios_three.append(cost_three / optimum)

    assert max(ratios_two) <= 2.0, "2-opt stays within the classic factor-2 bound"
    assert max(ratios_three) <= max(ratios_two) + 1e-12
    assert min(ratios_two) >= 1.0 - 1e-12 and min(ratios_three) >= 1.0 - 1e-12

    # --- Collinear cities: closed-form optimum and 2-opt reaches it ---------------------
    for count, spread in ((6, 10), (7, 20), (5, 3)):
        line = [(float(3 * i * spread // count), 0.0) for i in range(count)]
        optimum, _ = brute_force_optimal(line)
        assert abs(optimum - 2.0 * (max(x for (x, _) in line) - min(x for (x, _) in line))) <= 1e-9, \
            "the optimal closed tour on a line goes to one end and back"
        from_nn = two_opt_tour(line, nn_tour(line, start=count // 2))
        assert abs(tour_cost(line, from_nn) - optimum) <= 1e-9, \
            "2-opt untangles every crossing on a line"
        assert abs(tour_cost(line, three_opt_tour(line, from_nn)) - optimum) <= 1e-9

    # --- Interior start and duplicate cities --------------------------------------------
    dupes = [(0.0, 0.0), (0.0, 0.0), (5.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)]
    assert abs(brute_force_optimal(dupes)[0] - 20.0) <= 1e-9, "rectangle perimeter with doubles"
    dup_nn = nn_tour(dupes, start=2)
    dup_two = two_opt_tour(dupes, dup_nn)
    dup_three = three_opt_tour(dupes, dup_two)
    assert abs(tour_cost(dupes, dup_three) - 20.0) <= 1e-9
    assert tour_cost(dupes, dup_three) <= tour_cost(dupes, dup_nn) + 1e-9

    # --- Larger instance: heuristics only, deterministic -------------------------------
    big = lcg_points(40, 777, limit=100)
    big_nn = nn_tour(big)
    big_two = two_opt_tour(big, big_nn)
    big_three = three_opt_tour(big, big_two)
    assert is_tour(big_two, 40) and is_tour(big_three, 40)
    cost_nn = tour_cost(big, big_nn)
    cost_two = tour_cost(big, big_two)
    cost_three = tour_cost(big, big_three)
    assert cost_three <= cost_two <= cost_nn + 1e-9
    assert cost_two < cost_nn, "2-opt must find a real improvement on a 40-city zig-zag"
    assert cost_three < cost_two, "3-opt must find a real improvement over 2-opt here"
    assert cost_two >= mst_weight(big) - 1e-9
    assert big_two == two_opt_tour(big, big_nn) and big_three == three_opt_tour(big, big_two), \
        "repeated runs are identical"


if __name__ == "__main__":
    main()
    print("PASS")
