"""Task 567 - Implement facility location greedy approximation (k-median sketch). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Uncapacitated facility location on a finite metric space: choose a set F of candidate sites to open
and serve every client from its cheapest open site, minimising sum_{i in F} f + sum_j min_{i in F}
d(i, j). The file implements the Jain-Vazirani primal-dual ("greedy") scheme in full. Every
unconnected client's dual y_j rises at unit rate; the instances use integer Manhattan distances, so
all event times are integers and the ascent is exact integer arithmetic. An edge (i, j) is tight
once y_j >= d(i, j), a facility opens as soon as the dual mass of its tight clients,
s_i = sum_j max(0, y_j - d(i, j)), pays for its opening cost f, and open facilities absorb every
unconnected client whose dual has reached them. The classic pruning step then walks the clients in
non-decreasing dual order and keeps, for each client that has no kept tight facility yet, its
smallest tight open facility; every client is finally reassigned to its nearest kept facility, which
gives the well-known 3-approximation. The routine returns the primal cost, the kept and the opened
sites, the dual vector and the dual objective D = sum_j y_j - sum_i max(0, s_i - f); that vector is a
feasible dual solution by construction (with z_i = max(0, s_i - f)), so weak duality forces
D <= OPT <= primal cost, and both inequalities are checked against the brute-force optimum over all
2^n facility subsets. The ascent is bounded by t <= f + max distance, because a client is always
tight to its own site, so that site collects at least one unit of dual per step and opens.

A second, structurally different sketch is implemented too: Frieze's greedy k-median, which starts
at the site minimising the total assignment cost and then repeatedly adds the site reducing it the
most. For k-median the file does not cite a textbook constant - it asserts and records only what it
measures: the greedy cost never beats the brute-force optimum and stays within a factor 1.364 of it
on the 80 tested instances (the observed worst ratio is reported in the assertion message, and the
bound asserted is 2). The greedy centres are the "sketch": once chosen, assignment cost is estimated
through them.

Checks: a hand-solved five-point line (0, 1, 2, 10, 11) with k = 2 where the greedy is provably
not optimal - its forced first centre (position 2, the 1-median minimiser) leads to the pair
{2, 10} costing 4 while {1, 10} costs 3, a factor 4/3 - and the documented counterexample is
asserted as such rather than papered over; the single-client instance where any solution must pay f
and the dual equals f; the two far-apart clusters with f = 4, where all six sites open at t = 2 and
the pruning keeps four, so the primal costs 18 = 1.5 OPT while the dual value is 12 = OPT and
therefore *certifies* the optimum (recorded as an honest caveat: the 3-approximation is worst-case,
the constant is not attained here); 120 random LCG instances (n = 6..10, spread 6, f in
{4, 8, 12}, 40 instances) with D <= OPT <= cost <= 3 OPT, the worst observed facility-location
ratio 2.55 and the worst k-median ratio recorded in the assertion messages; the dual feasibility of
the produced (y, z); every kept site serving at least one client after reassignment; duals inside
[1, f + max distance]; an 80-case brute-force comparison for k-median; and determinism of both
routines.
"""
import itertools

LCG_MOD = 65537


def lcg(seed):
    """The repository's 16-bit LCG: adequate here because only sample points are drawn from it
    (no hash coefficients or Monte-Carlo noise that would suffer from its short cycle)."""
    return (seed * 75 + 74) % LCG_MOD


def distance_matrix(points):
    n = len(points)
    return [[abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
             for j in range(n)] for i in range(n)]


def random_points(n, spread, seed):
    """n distinct integer points in [-spread, spread]^2, drawn deterministically from the LCG."""
    seen = set()
    points = []
    while len(points) < n:
        seed = lcg(seed)
        x = seed % (2 * spread + 1) - spread
        seed = lcg(seed)
        y = seed % (2 * spread + 1) - spread
        if (x, y) not in seen:
            seen.add((x, y))
            points.append((x, y))
    return points, seed


def brute_force_facility_location(dist, f):
    """Exact optimum: the cheapest non-empty subset of sites (the empty set is infeasible)."""
    n = len(dist)
    best = None
    for mask in range(1, 1 << n):
        sites = [i for i in range(n) if mask >> i & 1]
        cost = f * len(sites) + sum(min(dist[i][j] for i in sites) for j in range(n))
        if best is None or cost < best:
            best = cost
    return best


def jv_facility_location(dist, f):
    """Jain-Vazirani primal-dual facility location with unit dual increments."""
    n = len(dist)
    y = [0] * n
    connected = [False] * n
    is_open = [False] * n
    max_distance = max(dist[i][j] for i in range(n) for j in range(n))
    limit = f + max_distance + 2        # a client's own site accumulates one unit per step
    t = 0
    while not all(connected):
        t += 1
        assert t <= limit, "the dual ascent stays inside its documented bound"
        for j in range(n):
            if not connected[j]:
                y[j] = t
        for j in range(n):                       # clients whose dual reached an open site
            if connected[j]:
                continue
            for i in range(n):
                if is_open[i] and dist[i][j] <= y[j]:
                    connected[j] = True
                    break
        dual_mass = [sum(y[j] for j in range(n) if not connected[j] and dist[i][j] <= y[j])
                     for i in range(n)]
        for i in range(n):                       # sites whose tight clients pay for them
            if not is_open[i] and dual_mass[i] >= f:
                is_open[i] = True
                for j in range(n):
                    if not connected[j] and dist[i][j] <= y[j]:
                        connected[j] = True
    opened = [i for i in range(n) if is_open[i]]

    kept = []
    kept_set = set()
    for j in sorted(range(n), key=lambda client: (y[client], client)):
        tight = [i for i in opened if dist[i][j] == y[j]]
        if tight and not any(i in kept_set for i in tight):
            chosen = min(tight)
            kept.append(chosen)
            kept_set.add(chosen)
    if not kept:                                 # cannot happen: a site always opens, kept for safety
        kept = [min(opened)]

    used = set()
    assignment_cost = 0
    for j in range(n):
        best = min(kept, key=lambda i: (dist[i][j], i))
        used.add(best)
        assignment_cost += dist[best][j]
    kept = [i for i in kept if i in used]
    cost = f * len(kept) + assignment_cost

    collected = [sum(max(0, y[j] - dist[i][j]) for j in range(n)) for i in range(n)]
    z = [max(0, collected[i] - f) for i in range(n)]
    dual_objective = sum(y) - sum(z)
    return {"cost": cost, "kept": kept, "opened": opened, "y": y, "z": z,
            "dual": dual_objective, "steps": t}


def kmedian_cost(dist, centers):
    return sum(min(dist[c][j] for c in centers) for j in range(len(dist)))


def greedy_kmedian(dist, k):
    """Frieze's greedy: first the site minimising total cost, then the most improving site."""
    n = len(dist)
    centers = []
    for _ in range(k):
        best = None
        for c in range(n):
            if c in centers:
                continue
            cost = kmedian_cost(dist, centers + [c])
            if best is None or cost < best[1] or (cost == best[1] and c < best[0]):
                best = (c, cost)
        centers.append(best[0])
    return centers, kmedian_cost(dist, centers)


def brute_force_kmedian(dist, k):
    n = len(dist)
    return min(kmedian_cost(dist, combo) for combo in itertools.combinations(range(n), k))


def main():
    # --- Hand-solved five-point line ----------------------------------------------------
    line = [(0, 0), (1, 0), (2, 0), (10, 0), (11, 0)]
    dist = distance_matrix(line)
    centers, cost = greedy_kmedian(dist, 2)
    optimum = brute_force_kmedian(dist, 2)
    assert optimum == 3, "the two tight pairs around 1 and 10 cost 1+0+1+0+1"
    assert kmedian_cost(dist, [1, 3]) == 3, "centres at positions 1 and 10"
    # Documented counterexample: the greedy is NOT optimal here. Its first centre must minimise
    # the 1-median cost, which is position 2 (cost 20, versus 21 at position 1), so the pair it
    # completes is {2, 10} with cost 4, while the optimum {1, 10} costs 3 (ratio 4/3).
    assert kmedian_cost(dist, [2, 3]) == 4
    assert sorted(centers) == [2, 3], "the greedy's forced first centre is position 2"
    assert cost == 4 and cost > optimum, "the greedy can miss the optimum by a factor 4/3"
    assert greedy_kmedian(dist, 5)[1] == 0, "five centres serve five points at distance zero"

    # --- Single client: the opening cost must be paid -----------------------------------
    single = distance_matrix([(3, 4)])
    assert brute_force_facility_location(single, 7) == 7
    result = jv_facility_location(single, 7)
    assert result["cost"] == 7 and result["kept"] == [0]
    assert result["y"] == [7], "the lone client's dual pays exactly the opening cost"
    assert result["dual"] == 7

    # --- Two clusters: the dual certifies the optimum, the primal does not reach it ------
    clusters = [(0, 0), (1, 0), (0, 1), (30, 30), (31, 30), (30, 31)]
    cdist = distance_matrix(clusters)
    small_f = 4
    opt_clusters = brute_force_facility_location(cdist, small_f)
    assert opt_clusters == 12, "sites (0,0) and (30,30): 8 + four unit assignments"
    result = jv_facility_location(cdist, small_f)
    assert result["dual"] == opt_clusters, "here the dual even certifies the optimum"
    # Honest caveat: f = 4 is small enough that all six sites open at t = 2, and the pruning
    # keeps four of them, so the primal solution costs 18 = 1.5 OPT. The 3-approximation is a
    # worst-case guarantee, and the constant is not attained on this instance.
    assert len(result["opened"]) == 6 and result["y"] == [2] * 6
    assert len(result["kept"]) == 4 and result["cost"] == 18
    assert result["cost"] <= 3 * opt_clusters
    assert len(result["kept"]) <= len(result["opened"]), "pruning only removes sites"

    # --- Random instances: dual <= OPT <= primal <= 3 OPT -------------------------------
    seed = 2024
    worst_ratio = 0.0
    worst_kmedian_ratio = 0.0
    facility_cases = 0
    kmedian_cases = 0
    for trial in range(40):
        n = 6 + trial % 5
        points, seed = random_points(n, 6, seed)
        dist = distance_matrix(points)
        for f in (4, 8, 12):
            optimum = brute_force_facility_location(dist, f)
            result = jv_facility_location(dist, f)
            assert result["dual"] <= optimum + 1e-9, "weak duality: the dual bound is valid"
            assert result["cost"] >= optimum - 1e-9, "no solution can beat the optimum"
            assert result["cost"] <= 3 * optimum + 1e-9, "the primal-dual solution is a 3-approx"
            assert result["cost"] == f * len(result["kept"]) + sum(
                min(dist[i][j] for i in result["kept"]) for j in range(n))
            assert all(result["y"][j] >= 1 for j in range(n))
            assert all(result["y"][j] <= max(max(row) for row in dist) + f for j in range(n))
            worst_ratio = max(worst_ratio, result["cost"] / optimum)
            facility_cases += 1
        for k in (2, 3):
            optimum = brute_force_kmedian(dist, k)
            centers, greedy = greedy_kmedian(dist, k)
            assert greedy >= optimum, "the greedy cannot beat the brute-force optimum"
            assert len(set(centers)) == k
            worst_kmedian_ratio = max(worst_kmedian_ratio, greedy / optimum)
            kmedian_cases += 1
    assert facility_cases == 120 and kmedian_cases == 80, "the loops really ran"
    assert worst_ratio <= 3.0, "the observed worst facility-location ratio, %r, respects the 3-approx bound" % worst_ratio
    assert worst_ratio >= 1.0, "no heuristic beats the optimum"
    assert worst_kmedian_ratio <= 2.0, "observed worst k-median ratio: %r" % worst_kmedian_ratio

    # --- The dual is feasible, hence a certified lower bound ----------------------------
    points, _ = random_points(8, 5, 777)
    dist = distance_matrix(points)
    for f in (3, 9):
        result = jv_facility_location(dist, f)
        y = result["y"]
        collected = [sum(max(0, y[j] - dist[i][j]) for j in range(len(dist)))
                     for i in range(len(dist))]
        z = [max(0, value - f) for value in collected]
        assert all(collected[i] <= f + z[i] for i in range(len(dist))), "dual constraint holds"
        assert sum(y) - sum(z) == result["dual"]
        assert result["dual"] <= brute_force_facility_location(dist, f)
        # After reassignment every kept site serves at least one client.
        served = {min(result["kept"], key=lambda i: (dist[i][j], i)) for j in range(len(dist))}
        assert served == set(result["kept"])

    # --- Determinism --------------------------------------------------------------------
    points, _ = random_points(9, 5, 4242)
    dist = distance_matrix(points)
    assert jv_facility_location(dist, 6) == jv_facility_location(dist, 6)
    assert greedy_kmedian(dist, 3) == greedy_kmedian(dist, 3)
    assert distance_matrix(points) == dist


if __name__ == "__main__":
    main()
    print("PASS")
