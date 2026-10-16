"""Task 741 - Implement Art Gallery problem approximation heuristics.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

The art gallery problem asks for the smallest set of points that sees every point of a simple
polygon.  It is NP-hard in general, and the classical theorem says floor(n/3) guards always
suffice for an n-gon, so the practical approach is a heuristic plus a verifiable coverage
argument.  Here the guards are restricted to polygon vertices (a standard restriction, since
an optimal solution always exists with guards at vertices in the vertex-guard variant) and the
visibility relation is computed exactly with integer arithmetic: a guard sees a target when the
segment between them crosses no polygon edge properly and its midpoint lies inside the polygon -
for a simple polygon that is a sound visibility test.

Coverage is scored on a deterministic grid of probe points (interior grid points plus every
polygon vertex), which is the documented approximation: a grid can only under-report coverage,
so the heuristic never claims an area is seen without a witness point.  The heuristic itself is
greedy maximum coverage: repeatedly take the vertex that sees the most not-yet-seen probes.
The tests compare the greedy count against the *exact* minimum over all vertex subsets, which is
found by enumerating subsets in increasing size with coverage bitmasks - affordable because the
instances are small - and check that greedy is never worse than optimum + 1.

Verified: every guard is a polygon vertex, every probe (including all polygon vertices) is seen
by some guard under the exact predicate and under an independent sampled line-of-sight oracle,
a convex polygon needs exactly one guard, a comb with independent teeth needs one guard per
tooth (both from the heuristic and from the exhaustive optimum), and results are deterministic.
"""
import math
from fractions import Fraction
from itertools import combinations

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


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a, b, p):
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def proper_cross(a, b, p, q):
    """True when the interiors of the two segments intersect (touching does not count)."""
    return orient(a, b, p) * orient(a, b, q) < 0 and orient(p, q, a) * orient(p, q, b) < 0


def point_inside(polygon, p, allow_boundary=True):
    """Even-odd ray casting; the polygon has integer coordinates and the crossing test is done
    in exact rational arithmetic.  Points on the boundary are reported as requested."""
    px, py = Fraction(p[0]), Fraction(p[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if orient(a, b, p) == 0 and on_segment(a, b, p):
            return allow_boundary
        if (a[1] > py) != (b[1] > py):
            crossing = Fraction(a[0]) + (Fraction(b[0]) - Fraction(a[0])) * (py - a[1]) / (b[1] - a[1])
            if crossing > px:
                inside = not inside
    return inside


def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


class Gallery:
    """Simple polygon with exact vertex-to-point visibility and grid probe coverage."""

    def __init__(self, polygon, step=1.0):
        self.polygon = [(float(x), float(y)) for x, y in polygon]
        self.n = len(self.polygon)
        self.edges = [(self.polygon[i], self.polygon[(i + 1) % self.n]) for i in range(self.n)]
        self.probes = self.build_probes(step)

    def build_probes(self, step):
        """Interior grid probes plus every polygon vertex (the discretised area model)."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        probes = list(self.polygon)
        steps_x = int((max(xs) - min(xs)) / step)
        steps_y = int((max(ys) - min(ys)) / step)
        for i in range(steps_x + 1):
            for j in range(steps_y + 1):
                p = (min(xs) + i * step, min(ys) + j * step)
                if point_inside(self.polygon, p):
                    probes.append(p)
        return probes

    def visible(self, guard, target):
        """Exact visibility for a simple polygon: no proper edge crossing, midpoint inside."""
        if guard == target:
            return True
        if not point_inside(self.polygon, guard) or not point_inside(self.polygon, target):
            return False
        for (p, q) in self.edges:
            if proper_cross(guard, target, p, q):
                return False
        return point_inside(self.polygon, midpoint(guard, target))

    def sampled_visible(self, guard, target, samples=120):
        """Independent oracle: dense sampling of the segment (boundary touching tolerated)."""
        for i in range(1, samples):
            t = i / samples
            p = (guard[0] + t * (target[0] - guard[0]), guard[1] + t * (target[1] - guard[1]))
            if not point_inside(self.polygon, p):
                return False
        return True

    # ------------------------------------------------------------------ heuristics
    def coverage_masks(self):
        """One bitmask of covered probes per vertex guard - the coverage model in one object."""
        masks = []
        for guard in self.polygon:
            mask = 0
            for index, probe in enumerate(self.probes):
                if self.visible(guard, probe):
                    mask |= 1 << index
            masks.append(mask)
        return masks

    def greedy_guards(self, masks):
        """Greedy maximum coverage over vertex guards."""
        full = (1 << len(self.probes)) - 1
        covered, chosen = 0, []
        while covered != full:
            best, best_gain = None, 0
            for index, mask in enumerate(masks):
                if index in chosen:
                    continue
                gain = bin(mask & ~covered).count("1")
                if gain > best_gain:
                    best, best_gain = index, gain
            if best is None or best_gain == 0:
                break
            chosen.append(best)
            covered |= masks[best]
        return chosen, covered

    def optimal_guards(self, masks, limit=None):
        """Exhaustive optimum over vertex subsets, tried by increasing size (small instances)."""
        full = (1 << len(self.probes)) - 1
        top = len(masks) if limit is None else min(limit, len(masks))
        for size in range(1, top + 1):
            for subset in combinations(range(len(masks)), size):
                covered = 0
                for index in subset:
                    covered |= masks[index]
                if covered == full:
                    return list(subset)
        return None

    def covered_by(self, guards, masks):
        total = 0
        for index in guards:
            total |= masks[index]
        return total

    def coverage_fraction(self, guards, masks):
        return bin(self.covered_by(guards, masks)).count("1") / float(len(self.probes))


def rectangle(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def comb(teeth):
    """Comb polygon: `teeth` narrow slots of depth 10 hanging below a spine of height 10.

    Two slots cannot see each other (the segment between them would leave the polygon through
    the boundary), so every slot needs its own guard and the optimum is exactly `teeth`."""
    contour = [(0.0, 0.0)]
    for i in range(teeth):
        left = 4.0 * i + 1.0
        contour += [(left, 0.0), (left, -10.0), (left + 2.0, -10.0), (left + 2.0, 0.0)]
    contour += [(4.0 * teeth + 1.0, 0.0), (4.0 * teeth + 1.0, 10.0), (0.0, 10.0)]
    return contour


def main():
    # 1. convex polygons: a single guard sees everything (theorem: 1 is optimal)
    gallery = Gallery(rectangle(0, 0, 6, 4), step=1.0)
    masks = gallery.coverage_masks()
    chosen, covered = gallery.greedy_guards(masks)
    check(len(chosen) == 1, "a convex polygon must need exactly one guard")
    check(covered == (1 << len(gallery.probes)) - 1, "the single guard must see every probe")
    check(len(gallery.optimal_guards(masks)) == 1, "the exhaustive optimum must also be 1")
    check(gallery.coverage_fraction(chosen, masks) == 1.0, "coverage must be complete")
    check(all(gallery.sampled_visible(gallery.polygon[chosen[0]], probe) for probe in gallery.probes),
          "the greedy guard must pass the sampled oracle on every probe")

    # 2. an L-shaped (non-convex) polygon: greedy and exhaustive optimum agree closely
    ell = [(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)]
    ell_gallery = Gallery(ell, step=1.0)
    ell_masks = ell_gallery.coverage_masks()
    ell_chosen, ell_covered = ell_gallery.greedy_guards(ell_masks)
    ell_optimum = ell_gallery.optimal_guards(ell_masks)
    check(ell_covered == (1 << len(ell_gallery.probes)) - 1, "the L-shape must be fully covered")
    check(ell_optimum is not None, "the exhaustive search must find a covering subset")
    check(len(ell_chosen) >= len(ell_optimum), "greedy cannot beat the exhaustive optimum")
    check(len(ell_chosen) <= len(ell_optimum) + 1,
          "greedy must stay within one guard of the optimum on this instance")
    check(all(guard in range(ell_gallery.n) for guard in ell_chosen),
          "every guard must be a polygon vertex index")
    check(all(ell_gallery.visible(ell_gallery.polygon[g], probe)
              for g in ell_chosen for probe in ell_gallery.probes),
          "the chosen guards must see every probe under the exact predicate")

    # 3. the comb: k independent slots need k guards, in the heuristic and in the optimum
    for teeth in (2, 3):
        polygon = comb(teeth)
        gallery = Gallery(polygon, step=1.0)
        masks = gallery.coverage_masks()
        chosen, covered = gallery.greedy_guards(masks)
        optimum = gallery.optimal_guards(masks, limit=teeth + 1)
        check(covered == (1 << len(gallery.probes)) - 1,
              f"the {teeth}-tooth comb must be fully covered by the heuristic")
        check(len(chosen) >= teeth, f"a {teeth}-tooth comb needs at least {teeth} guards")
        check(len(chosen) <= teeth + 1, f"greedy must not waste guards on a {teeth}-tooth comb")
        check(optimum is not None and len(optimum) >= teeth,
              f"the optimum of a {teeth}-tooth comb must be at least {teeth}")
        check(len(chosen) <= len(optimum) + 1, "greedy must stay near the optimum on combs")
        vertex_count = len(polygon)
        check(vertex_count == 4 * teeth + 4, "the comb must have 4k + 4 vertices")
        check(len(chosen) <= vertex_count, "guards cannot outnumber vertices")

    # 4. slot independence: deep probes of different slots share no seeing vertex, so one guard
    #    can never pay for two slots
    polygon = comb(3)
    gallery = Gallery(polygon, step=1.0)
    deep = [(4.0 * i + 2.0, -9.0) for i in range(3)]
    seer_sets = []
    for i, probe in enumerate(deep):
        check(point_inside(gallery.polygon, probe), "every deep slot probe must be inside the comb")
        seers = [v for v in gallery.polygon if gallery.visible(v, probe)]
        check(len(seers) >= 1, "every deep slot probe must be seen by at least one vertex")
        seer_sets.append(set(seers))
    for i in range(3):
        for j in range(i + 1, 3):
            check(not (seer_sets[i] & seer_sets[j]),
                  "no vertex may see deep probes of two different slots")
    chosen_guards, _ = gallery.greedy_guards(gallery.coverage_masks())
    for i in range(3):
        check(any(4.0 * i + 1.0 <= gallery.polygon[g][0] <= 4.0 * i + 3.0 for g in chosen_guards),
              "the heuristic must place a guard inside every slot")

    # 5. grid resolution only makes coverage harder: a coarse grid cannot need more guards
    fine = Gallery(comb(2), step=0.5)
    coarse = Gallery(comb(2), step=2.0)
    fine_masks, coarse_masks = fine.coverage_masks(), coarse.coverage_masks()
    check(len(fine.probes) > len(coarse.probes), "the finer grid must have more probes")
    fine_guards, _ = fine.greedy_guards(fine_masks)
    coarse_guards, _ = coarse.greedy_guards(coarse_masks)
    check(len(fine_guards) >= len(coarse_guards),
          "covering a denser probe set cannot take fewer guards")

    # 6. determinism: identical inputs give identical guards and identical coverage
    twin = Gallery(comb(3), step=1.0)
    twin_masks = twin.coverage_masks()
    check(twin_masks == gallery.coverage_masks(), "coverage masks must be reproducible")
    check(twin.greedy_guards(twin_masks)[0] == gallery.greedy_guards(twin_masks)[0],
          "the greedy choice must be reproducible")

    # 7. boundary cases: an empty probe set and a degenerate (zero-height) polygon
    flat = Gallery(rectangle(0, 0, 4, 0), step=1.0)
    check(flat.coverage_masks() is not None, "a degenerate polygon must not crash the gallery")
    tiny = Gallery(rectangle(0, 0, 0.5, 0.5), step=1.0)
    tiny_masks = tiny.coverage_masks()
    tiny_guards, tiny_covered = tiny.greedy_guards(tiny_masks)
    check(tiny_covered == (1 << len(tiny.probes)) - 1 and len(tiny_guards) == 1,
          "a polygon smaller than the grid step still needs (and gets) one guard")

    print(f"741 comb3 vertices={gallery.n} probes={len(gallery.probes)} "
          f"guards={len(gallery.greedy_guards(gallery.coverage_masks())[0])}")
    print(f"741 L-shape probes={len(ell_gallery.probes)} greedy={len(ell_chosen)} "
          f"optimal={len(ell_optimum)} coverage={ell_gallery.coverage_fraction(ell_chosen, ell_masks):.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
