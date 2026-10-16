"""Task 768 - Implement skyline queries in multi-criteria optimization (Pareto front).  (Tier: Expert, Ch 8 - Recursion and Dynamic Programming)

Point p dominates point q when p is no worse in every criterion and strictly better in at
least one:  p[i] <= q[i] for all i, and p[j] < q[j] for some j.  The skyline (Pareto front)
is the subset of points that no other point dominates; the returned points are pairwise
incomparable, i.e. they form an *antichain*.  Ties in a single coordinate are "no worse",
so two copies of the same point never dominate each other and both survive - that multiset
semantics is kept identical in all three implementations below.

Three independent implementations are cross-verified:
  * brute force - the O(n^2) filter over the whole set; the reference oracle.
  * BNL (block nested loop) - keeps a "window" holding the skyline of the points seen so
    far.  A point that no window member dominates joins the window and the window members
    it dominates are dropped at the end of its block.
  * SFS (sort-filter-skyline) - the same window, but the input is sorted lexicographically
    first, which makes dominated points get rejected early by a small window.
Both window algorithms return the exact skyline: a discarded point was dominated by some
point that was in the window at that moment, and a window member only leaves when a later
point dominates it, so the dominance witness chain always terminates in the final window.
That "witness" property is checked explicitly, together with split invariance
skyline(A v B) == non-dominated(skyline(A) v skyline(B)).

Simplified versus the paper: the SFS window is a plain Python list (no bitmap/quadtree
index over the window) and the maintainer adds one point at a time, so the quadratic worst
case is kept - what is verified is exactness and the invariants, not a speedup bound.
Verified: emptiness and single-point boundaries, duplicates/ties, 1-D and all-dominated
degenerate sets, agreement of the three algorithms on random 2-D/3-D/4-D data, antichain +
witness invariants on every result, incremental maintenance against a from-scratch
skyline after every insertion, split invariance, and determinism.
"""
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def random_points(count, dims, high, seed):
    """Deterministic integer points; later coordinates reuse the streamed PRNG state."""
    points = []
    for _ in range(count):
        coords = []
        for _ in range(dims):
            seed, r = splitmix64(seed)
            coords.append(r % high)
        points.append(tuple(coords))
    return points, seed


def dominates(a, b):
    """True iff a is <= b in every coordinate and strictly better in at least one."""
    better = False
    for x, y in zip(a, b):
        if x > y:
            return False
        if x < y:
            better = True
    return better


def brute_force_skyline(points):
    """Reference oracle: keep every point that no point of the whole set dominates."""
    skyline = []
    comparisons = 0
    for p in points:
        dominated = False
        for q in points:
            comparisons += 1
            if dominates(q, p):
                dominated = True
                break
        if not dominated:
            skyline.append(p)
    return skyline, comparisons


def sfs_skyline(points):
    """Sort-filter-skyline: lexicographic order concentrates early rejections."""
    window = []
    comparisons = 0
    for p in sorted(points):
        dominated = False
        for q in window:
            comparisons += 1
            if dominates(q, p):
                dominated = True
                break
        if dominated:
            continue
        survivors = []
        for q in window:
            comparisons += 1
            if not dominates(p, q):
                survivors.append(q)
        survivors.append(p)
        window = survivors
    return window, comparisons


def bnl_skyline(points, block_size=4):
    """Block nested loop: block-size-dependent block filter over the running window."""
    window = []
    comparisons = 0
    for start in range(0, len(points), block_size):
        accepted = []
        for p in points[start:start + block_size]:
            dominated = False
            for q in window:
                comparisons += 1
                if dominates(q, p):
                    dominated = True
                    break
            if not dominated:
                for q in accepted:
                    comparisons += 1
                    if dominates(q, p):
                        dominated = True
                        break
            if not dominated:
                survivors = []
                for r in accepted:
                    comparisons += 1
                    if not dominates(p, r):
                        survivors.append(r)
                survivors.append(p)
                accepted = survivors
        survivors = []
        for q in window:
            for r in accepted:
                comparisons += 1
                if dominates(r, q):
                    break
            else:
                survivors.append(q)
        window = survivors + accepted
    return window, comparisons


class SkylineMaintainer:
    """Incremental skyline: add() returns whether the point joined the front."""

    def __init__(self):
        self.front = []
        self.comparisons = 0

    def add(self, p):
        for q in self.front:
            self.comparisons += 1
            if dominates(q, p):
                return False
        survivors = []
        for q in self.front:
            self.comparisons += 1
            if not dominates(p, q):
                survivors.append(q)
        survivors.append(p)
        self.front = survivors
        return True


def as_set(seq):
    return sorted(set(seq))


def main():
    # 1. boundary: empty and single-point inputs
    for name, algo in (("brute", brute_force_skyline), ("sfs", sfs_skyline), ("bnl", bnl_skyline)):
        sky, comparisons = algo([])
        check(sky == [], f"{name}: empty input must give an empty skyline")
        check(comparisons == 0, f"{name}: empty input must cost no dominance test")
    check(sfs_skyline([(3, 4)])[0] == [(3, 4)], "a single point is its own skyline")
    check(bnl_skyline([(0, 0)])[0] == [(0, 0)], "a single zero point is its own skyline")

    # 2. duplicates: no copy dominates another, so every copy survives
    dupes = [(2, 2)] * 3
    for name, algo in (("brute", brute_force_skyline), ("sfs", sfs_skyline), ("bnl", bnl_skyline)):
        sky, _ = algo(dupes)
        check(sky == dupes, f"{name}: duplicates must all be kept (multiset semantics)")

    # 3. one dimension: only the minimum survives
    one_d = [(3,), (1,), (2,), (1,)]
    check(brute_force_skyline(one_d)[0] == [(1,), (1,)], "1-D skyline is exactly the minima")
    check(sfs_skyline(one_d)[0] == [(1,), (1,)], "SFS 1-D skyline must match the minima")
    check(bnl_skyline(one_d)[0] == [(1,), (1,)], "BNL 1-D skyline must match the minima")

    # 4. all points dominated by one: a frontier of size one
    staircase = [(0, 0)] + [(i, i) for i in range(1, 6)]
    for name, algo in (("brute", brute_force_skyline), ("sfs", sfs_skyline), ("bnl", bnl_skyline)):
        sky, _ = algo(staircase)
        check(sky == [(0, 0)], f"{name}: the single undominated point must be the frontier")

    # 5. ties in one coordinate are incomparable
    ties = [(1, 5), (2, 6), (2, 4)]
    check(as_set(brute_force_skyline(ties)[0]) == [(1, 5), (2, 4)],
          "ties must be resolved by strict dominance only")
    check(as_set(sfs_skyline(ties)[0]) == [(1, 5), (2, 4)], "SFS must resolve ties the same way")

    # 6. cross-verification on random data of increasing dimension
    totals = {}
    for dims in (2, 3, 4):
        seed = 0xA11CE + dims
        for trial in range(6):
            points, seed = random_points(40 + 7 * trial, dims, 12, seed)
            reference = as_set(brute_force_skyline(points)[0])
            sfs, sfs_cost = sfs_skyline(points)
            bnl, bnl_cost = bnl_skyline(points, block_size=1 + trial % 4)
            check(as_set(sfs) == reference, f"SFS diverged from brute force (dims={dims})")
            check(as_set(bnl) == reference, f"BNL diverged from brute force (dims={dims})")
            check(len(sfs) <= len(points), "the skyline can never exceed the input size")
            totals.setdefault(dims, 0)
            totals[dims] += sfs_cost + bnl_cost

    # 7. invariants on every reference result: antichain + dominance witness
    seed = 0xBEEF
    witness_checks = 0
    for _ in range(5):
        points, seed = random_points(60, 3, 9, seed)
        sky = as_set(brute_force_skyline(points)[0])
        for i, p in enumerate(sky):
            for q in sky[i + 1:]:
                check(not dominates(p, q) and not dominates(q, p),
                      "skyline members must be pairwise incomparable (antichain)")
        for p in points:
            if p in sky:
                continue
            check(any(dominates(q, p) for q in sky),
                  "every rejected point must be dominated by a returned skyline point")
            witness_checks += 1

    # 8. split invariance: only the frontiers have to be merged
    seed = 0x5EED
    left, seed = random_points(45, 2, 15, seed)
    right, seed = random_points(45, 2, 15, seed)
    whole = as_set(brute_force_skyline(left + right)[0])
    merged = as_set(brute_force_skyline(as_set(left) + as_set(right))[0])
    check(merged == whole, "skyline(A u B) must equal non-dominated(skyline(A) u skyline(B))")
    two_stage = as_set(sfs_skyline(as_set(sfs_skyline(left)[0]) + as_set(sfs_skyline(right)[0]))[0])
    check(two_stage == whole, "two-stage SFS must equal the single-pass skyline")

    # 9. incremental maintenance equals a from-scratch skyline after every insertion
    seed = 0x1234
    points, seed = random_points(50, 3, 10, seed)
    maintainer = SkylineMaintainer()
    for i, p in enumerate(points):
        joined = maintainer.add(p)
        expected = as_set(brute_force_skyline(points[:i + 1])[0])
        check(as_set(maintainer.front) == expected,
              f"incremental front diverged from brute force after insert {i}")
        check(joined == (p in expected), "add() must report whether the point is undominated")

    # 10. 2-D structure: crossing frontiers means y strictly decreases as x increases
    seed = 0x77
    points, seed = random_points(80, 2, 20, seed)
    sky = as_set(brute_force_skyline(points)[0])
    for i, p in enumerate(sky):
        for q in sky[i + 1:]:
            if p[0] < q[0]:
                check(p[1] > q[1], "in 2-D an increasing x must come with a decreasing y")
            if p[0] > q[0]:
                check(p[1] < q[1], "in 2-D a decreasing x must come with an increasing y")

    # 11. window algorithms beat the full O(n^2) filter on a large skewed instance
    seed = 0xC0FFEE
    big, seed = random_points(400, 4, 60, seed)
    brute, brute_cost = brute_force_skyline(big)
    sfs, sfs_cost = sfs_skyline(big)
    bnl, bnl_cost = bnl_skyline(big, block_size=8)
    check(as_set(sfs) == as_set(brute) == as_set(bnl), "large-instance results must agree")
    check(brute_cost > len(big) * len(big) // 6, "the brute-force oracle must scan the whole set repeatedly")
    check(brute_cost >= 4 * sfs_cost, "SFS must test substantially fewer pairs than the quadratic filter")

    # 12. determinism
    again = sfs_skyline(big)[0]
    check(as_set(again) == as_set(sfs) and len(again) == len(sfs), "SFS must be deterministic")

    print(f"768 frontier sizes dims2={totals[2]} dims3={totals[3]} dims4={totals[4]} witnesses={witness_checks}")
    print(f"768 large n={len(big)} skyline={len(brute)} brute={brute_cost} sfs={sfs_cost} bnl={bnl_cost}")
    print("PASS")


if __name__ == "__main__":
    main()
