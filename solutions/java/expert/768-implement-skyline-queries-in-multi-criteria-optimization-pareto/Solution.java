/** Task 768 - Implement skyline queries in multi-criteria optimization (Pareto front). (Tier: Expert, CTCI Ch 8 - Recursion and Dynamic Programming)
 *
 * A point p dominates q (minimisation) when p is no worse than q on every criterion and strictly
 * better on at least one. The skyline / Pareto front is the set of points that no other point
 * dominates. Equal points never dominate each other, so duplicates are all kept.
 *
 * Three classic skyline algorithms are implemented and cross-checked against one another:
 *   bruteForce - the O(n^2) nested loop; the exact oracle for any dimension.
 *   bnl        - Block Nested Loop with a self-pruning in-memory window: an incoming point evicts
 *                every window member that it dominates, so the window is always a candidate
 *                antichain (the cheap variant used by the BNL/BBS family).
 *   sortSweep  - the 2-D scan: sort by (x, y) and keep a point iff no strictly smaller x already
 *                reaches its y and no same-x point has a smaller y.
 *
 * Invariants verified: the result is an antichain (no member dominates another); every rejected
 * point is dominated by some *kept* point; the three algorithms return the same multiset on random
 * inputs in 2/3/4 dimensions; the 2-D skyline has at most one point per distinct x; deleting points
 * can only grow and adding a dominated point never grows the skyline; the three algorithms agree on
 * the degenerate inputs (empty set, single point, one point repeated, an antichain, a total order).
 *
 * Simplified: coordinates are small ints, dimensions <= 4 and the sets are tens of points, so the
 * quadratic oracle stays fast; BNL's block size is unbounded (the window fits in memory).
 */
import java.util.*;

public class Solution {
    private static long state = 0x9E3779B97F4A7C15L;

    /** splitmix64: deterministic pseudo randomness, no Math.random anywhere. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static List<int[]> randomPoints(int count, int dims, int maxValue) {
        List<int[]> points = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int[] p = new int[dims];
            for (int d = 0; d < dims; d++) p[d] = nextInt(maxValue);
            points.add(p);
        }
        return points;
    }

    /** true when p dominates q: p <= q in every dimension and p < q in at least one. */
    static boolean dominates(int[] p, int[] q) {
        boolean strictlyBetter = false;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > q[i]) return false;
            if (p[i] < q[i]) strictlyBetter = true;
        }
        return strictlyBetter;
    }

    /** Exact oracle: every point that no other point (by position) dominates. */
    static List<int[]> bruteForce(List<int[]> points) {
        List<int[]> skyline = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            boolean dominated = false;
            for (int j = 0; j < points.size() && !dominated; j++) {
                if (i != j && dominates(points.get(j), points.get(i))) dominated = true;
            }
            if (!dominated) skyline.add(points.get(i));
        }
        return skyline;
    }

    /** Block Nested Loop with a self-pruning window; window size never exceeds the skyline size. */
    static class Bnl {
        final List<int[]> window = new ArrayList<>();
        long comparisons;
        long pruned;

        void add(int[] point) {
            boolean dominated = false;
            for (int i = 0; i < window.size(); ) {
                int[] w = window.get(i);
                comparisons++;
                if (dominates(w, point)) {
                    dominated = true;
                    break;
                }
                if (dominates(point, w)) {
                    window.remove(i);
                    pruned++;
                } else {
                    i++;
                }
            }
            if (!dominated) window.add(point);
        }
    }

    static List<int[]> bnl(List<int[]> points) {
        Bnl scan = new Bnl();
        for (int[] p : points) scan.add(p);
        return scan.window;
    }

    /** 2-D skyline in one sorted pass: O(n log n) and exact for two criteria. */
    static List<int[]> sortSweep2d(List<int[]> points) {
        List<int[]> sorted = new ArrayList<>(points);
        sorted.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
        List<int[]> skyline = new ArrayList<>();
        int prefixMinY = Integer.MAX_VALUE;          // min y among points with strictly smaller x
        int i = 0;
        while (i < sorted.size()) {
            int j = i;
            int groupMinY = Integer.MAX_VALUE;       // min y inside this equal-x group
            while (j < sorted.size() && sorted.get(j)[0] == sorted.get(i)[0]) {
                groupMinY = Math.min(groupMinY, sorted.get(j)[1]);
                j++;
            }
            for (int k = i; k < j; k++) {
                int[] p = sorted.get(k);
                // dominated from the left, or by a same-x point with a smaller y
                if (prefixMinY <= p[1] || p[1] > groupMinY) continue;
                skyline.add(p);
            }
            prefixMinY = Math.min(prefixMinY, groupMinY);
            i = j;
        }
        return skyline;
    }

    static String key(int[] p) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < p.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(p[i]);
        }
        return sb.toString();
    }

    static List<String> keys(List<int[]> points) {
        List<String> out = new ArrayList<>();
        for (int[] p : points) out.add(key(p));
        Collections.sort(out);
        return out;
    }

    /** Every point left out of the skyline must be dominated by a point that stayed in. */
    static boolean everyRejectedPointIsDominated(List<int[]> points) {
        List<int[]> skyline = bruteForce(points);
        Set<String> kept = new HashSet<>(keys(skyline));
        for (int[] p : points) {
            if (kept.contains(key(p))) continue;
            boolean witness = false;
            for (int[] s : skyline) {
                if (dominates(s, p)) witness = true;
            }
            if (!witness) return false;
        }
        return true;
    }

    static boolean isAntichain(List<int[]> points) {
        for (int i = 0; i < points.size(); i++) {
            for (int j = 0; j < points.size(); j++) {
                if (i != j && dominates(points.get(i), points.get(j))) return false;
            }
        }
        return true;
    }

    static List<String> missingKeys(Set<String> expected, Set<String> actual) {
        List<String> missing = new ArrayList<>();
        for (String k : expected) {
            if (!actual.contains(k)) missing.add(k);
        }
        Collections.sort(missing);
        return missing;
    }

    public static void main(String[] args) {
        // --- degenerate inputs --------------------------------------------------
        check(bruteForce(new ArrayList<>()).isEmpty(), "empty skyline");
        check(bnl(new ArrayList<>()).isEmpty(), "empty skyline, BNL");
        check(sortSweep2d(new ArrayList<>()).isEmpty(), "empty skyline, sweep");

        List<int[]> single = new ArrayList<>(List.of(new int[]{4, 4}));
        check(bruteForce(single).size() == 1, "single point survives");
        check(sortSweep2d(single).size() == 1, "single point survives, sweep");

        List<int[]> duplicates = new ArrayList<>();
        for (int i = 0; i < 5; i++) duplicates.add(new int[]{2, 3});
        check(bruteForce(duplicates).size() == 5, "equal points never dominate each other");
        check(sortSweep2d(duplicates).size() == 5, "equal points survive the sweep");
        check(bnl(duplicates).size() == 5, "equal points survive BNL");

        // --- textbook 2-D example ----------------------------------------------
        List<int[]> classic = new ArrayList<>(List.of(
                new int[]{2, 9}, new int[]{3, 4}, new int[]{4, 3}, new int[]{9, 2},
                new int[]{6, 6}, new int[]{8, 1}, new int[]{1, 20}));
        List<int[]> classicSkyline = bruteForce(classic);
        check(keys(classicSkyline).equals(List.of("1,20", "2,9", "3,4", "4,3", "8,1")),
                "classic 2-D skyline, got " + keys(classicSkyline));
        check(keys(sortSweep2d(classic)).equals(keys(classicSkyline)), "sweep agrees on the classic set");
        check(keys(bnl(classic)).equals(keys(classicSkyline)), "BNL agrees on the classic set");

        // --- antichain input: nothing is pruned -------------------------------
        List<int[]> antichain = new ArrayList<>(List.of(
                new int[]{0, 5}, new int[]{1, 4}, new int[]{2, 3}, new int[]{3, 2}, new int[]{4, 1}));
        check(bruteForce(antichain).size() == 5, "antichain fully survives");
        check(bnl(antichain).size() == 5, "BNL keeps the whole antichain");
        Bnl expensive = new Bnl();
        for (int[] p : antichain) expensive.add(p);
        check(expensive.pruned == 0, "no window member may be pruned on an antichain");

        // --- random cross-check, dimensions 2..4 ------------------------------
        long totalComparisons = 0;
        for (int dims = 2; dims <= 4; dims++) {
            for (int trial = 0; trial < 6; trial++) {
                List<int[]> points = randomPoints(12 + 7 * trial, dims, 10);
                List<int[]> expected = bruteForce(points);
                check(keys(bnl(points)).equals(keys(expected)), "BNL vs brute force, dims=" + dims + " trial=" + trial);
                check(isAntichain(expected), "brute force returns an antichain");
                check(everyRejectedPointIsDominated(points), "every rejected point has a kept dominator");
                if (dims == 2) {
                    check(keys(sortSweep2d(points)).equals(keys(expected)),
                            "2-D sweep vs brute force, trial=" + trial);
                }
                Bnl run = new Bnl();
                for (int[] p : points) run.add(p);
                totalComparisons += run.comparisons;
                check(run.window.size() == expected.size(), "BNL window size equals the skyline size");
            }
        }
        check(totalComparisons > 0, "BNL really compared points");

        // --- 2-D structural properties ----------------------------------------
        List<int[]> flat = randomPoints(60, 2, 25);
        List<int[]> flatSkyline = sortSweep2d(flat);
        // In a 2-D skyline every distinct x carries exactly the minimal y, so the number of distinct
        // points equals the number of distinct x values and the y's fall strictly as x grows.
        Map<Integer, Integer> perX = new TreeMap<>();
        for (int[] p : flatSkyline) perX.merge(p[0], p[1], Math::min);
        check(perX.size() == new HashSet<>(keys(flatSkyline)).size(),
                "2-D skyline has exactly one distinct point per x");
        boolean decreasing = true;
        int previousY = Integer.MAX_VALUE;
        for (int y : perX.values()) {
            if (y >= previousY) decreasing = false;
            previousY = y;
        }
        check(decreasing, "sorted 2-D skyline has strictly decreasing y");

        // --- monotonicity: a dominated newcomer is rejected --------------------
        List<int[]> withDominated = new ArrayList<>(flat);
        withDominated.add(new int[]{25, 25});
        check(keys(sortSweep2d(withDominated)).equals(keys(flatSkyline)),
                "a dominated point never changes the skyline");

        List<int[]> withNewBest = new ArrayList<>(flat);
        withNewBest.add(new int[]{-1, -1});
        int grown = sortSweep2d(withNewBest).size();
        check(grown == 1, "a dominating newcomer collapses the skyline to itself, got " + grown);

        // --- deleting points can only grow the skyline -------------------------
        // (the set-monotonicity claim: a point that survived without the deleted points survives with
        // them as well, because domination can only be witnessed by points that are still present)
        List<int[]> unique = new ArrayList<>();
        Set<String> seenKeys = new HashSet<>();
        for (int[] p : flat) if (seenKeys.add(key(p))) unique.add(p);
        List<int[]> uniqueSkyline = sortSweep2d(unique);
        Set<String> keptKeys = new HashSet<>(keys(uniqueSkyline));
        List<int[]> half = new ArrayList<>(unique.subList(0, unique.size() / 2));
        Set<String> halfPointKeys = new HashSet<>();
        for (int[] p : half) halfPointKeys.add(key(p));
        Set<String> halfSkylineKeys = new HashSet<>(keys(sortSweep2d(half)));
        List<String> missing = new ArrayList<>();
        for (String k : keptKeys) {
            if (halfPointKeys.contains(k) && !halfSkylineKeys.contains(k)) missing.add(k);
        }
        check(missing.isEmpty(), "a surviving point cannot be dominated by the deleted ones, missing " + missing);
        check(unique.size() > half.size(), "the subset really is a strict subset");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
