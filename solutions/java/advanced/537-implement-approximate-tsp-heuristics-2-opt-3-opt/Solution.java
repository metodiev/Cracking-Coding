/** Task 537 - Implement approximate TSP heuristics (2-opt / 3-opt). (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Euclidean TSP heuristics on small instances: a deterministic nearest-neighbour construction
 * (greedy from vertex 0, ties broken by the smallest index) followed by best-improvement local
 * search over the full cyclic 2-opt and 3-opt neighbourhoods. Both neighbourhoods are generated
 * from cuts of the tour cycle: 2-opt cuts the edges (t_a, t_{a+1}), (t_b, t_{b+1 mod n}) and
 * relinks the two segments in the single non-trivial way; 3-opt cuts three tour edges and tries
 * the seven non-identity reconnections (S2 S3r), (S2r S3), (S2r S3r), (S3r S2), (S3 S2r),
 * (S3 S2), (S3r S2r) of the two middle segments. Every candidate tour is re-anchored so that
 * vertex 0 comes first (rotation preserves the cost of a cycle), which makes the search result
 * a canonical array and keeps the deterministic tie-breaking reproducible.
 *
 * Documented checks: tours are permutations of 0..n-1 and costs are recomputed from the
 * distance matrix; each search only ever accepts a strictly improving move and both searches are
 * idempotent (re-running them changes nothing), while a separate verifier re-scans the whole 2-opt
 * neighbourhood and the full seven-case 3-opt neighbourhood to confirm the reported tours admit no
 * improving move; on the 8-point convex
 * instance (points on a circle) the 2-opt tour is optimal, matching the classic result that
 * any 2-optimal tour of points in convex position is optimal, and its cost equals the closed
 * form (the convex hull perimeter); on LCG point sets, 2-opt never loses to nearest neighbour,
 * 3-opt never loses to 2-opt, and both are bracketed by the brute-force optimum computed over
 * all (n-1)!/2 tours for n <= 8; the hand-checked unit square instance has optimum 4.
 */
import java.util.*;

public class Solution {
    static final double EPS = 1e-9;

    static double[][] distances(double[][] pts) {
        int n = pts.length;
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dx = pts[i][0] - pts[j][0];
                double dy = pts[i][1] - pts[j][1];
                d[i][j] = Math.sqrt(dx * dx + dy * dy);
            }
        }
        return d;
    }

    static double cycleCost(double[][] d, int[] tour) {
        double total = 0.0;
        for (int i = 0; i < tour.length; i++) {
            total += d[tour[i]][tour[(i + 1) % tour.length]];
        }
        return total;
    }

    /** Re-anchor the cycle so that vertex 0 sits at index 0 (a cyclic rotation only). */
    static int[] rotateToZero(int[] tour) {
        int n = tour.length;
        int at = 0;
        for (int i = 0; i < n; i++) {
            if (tour[i] == 0) at = i;
        }
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = tour[(at + i) % n];
        return out;
    }

    static int[] nearestNeighbor(double[][] d) {
        int n = d.length;
        boolean[] used = new boolean[n];
        int[] tour = new int[n];
        int current = 0;
        used[0] = true;
        tour[0] = 0;
        for (int step = 1; step < n; step++) {
            int best = -1;
            for (int v = 0; v < n; v++) {
                if (used[v]) continue;
                if (best < 0 || d[current][v] < d[current][best] - EPS) best = v;
            }
            tour[step] = best;
            used[best] = true;
            current = best;
        }
        return tour;
    }

    /** Full cyclic 2-opt move on the cuts (t_a, t_{a+1}) and (t_b, t_{b+1 mod n}). */
    static int[] apply2Opt(int[] tour, int a, int b) {
        int n = tour.length;
        int[] out = new int[n];
        int head = 0;
        int firstLength = n - (b - a);
        for (int k = 0; k < firstLength; k++) out[head++] = tour[(b + 1 + k) % n];
        for (int k = b; k > a; k--) out[head++] = tour[k];
        return out;
    }

    /** One of the seven non-identity 3-opt reconnections of the segments S2 and S3. */
    static int[] apply3Opt(int[] tour, int a, int b, int c, int variant) {
        int n = tour.length;
        int[] s1 = new int[n - (c - a)];
        for (int k = 0; k < s1.length; k++) s1[k] = tour[(c + 1 + k) % n];
        int[] s2 = new int[b - a];
        for (int k = 0; k < s2.length; k++) s2[k] = tour[a + 1 + k];
        int[] s3 = new int[c - b];
        for (int k = 0; k < s3.length; k++) s3[k] = tour[b + 1 + k];
        boolean flip2;
        boolean flip3;
        boolean swap;
        switch (variant) {
            case 1: flip2 = false; flip3 = true; swap = false; break;
            case 2: flip2 = true; flip3 = false; swap = false; break;
            case 3: flip2 = true; flip3 = true; swap = false; break;
            case 4: flip2 = false; flip3 = true; swap = true; break;
            case 5: flip2 = true; flip3 = false; swap = true; break;
            case 6: flip2 = false; flip3 = false; swap = true; break;
            default: flip2 = true; flip3 = true; swap = true; break;
        }
        int[] out = new int[n];
        int head = 0;
        for (int value : s1) out[head++] = value;
        int[][] order = swap ? new int[][]{s3, s2} : new int[][]{s2, s3};
        boolean[] flips = swap ? new boolean[]{flip3, flip2} : new boolean[]{flip2, flip3};
        for (int segment = 0; segment < 2; segment++) {
            int[] values = order[segment];
            if (flips[segment]) {
                for (int k = values.length - 1; k >= 0; k--) out[head++] = values[k];
            } else {
                for (int value : values) out[head++] = value;
            }
        }
        return out;
    }

    static double minTwoOptDelta(double[][] d, int[] tour) {
        int n = tour.length;
        double best = Double.POSITIVE_INFINITY;
        double current = cycleCost(d, tour);
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                if (b == a + 1) continue;
                best = Math.min(best, cycleCost(d, apply2Opt(tour, a, b)) - current);
            }
        }
        return best;
    }

    static double minThreeOptDelta(double[][] d, int[] tour) {
        int n = tour.length;
        double best = Double.POSITIVE_INFINITY;
        double current = cycleCost(d, tour);
        for (int a = 0; a < n - 2; a++) {
            for (int b = a + 1; b < n - 1; b++) {
                for (int c = b + 1; c < n; c++) {
                    for (int variant = 1; variant <= 7; variant++) {
                        if (c - b < 1) continue;
                        best = Math.min(best, cycleCost(d, apply3Opt(tour, a, b, c, variant)) - current);
                    }
                }
            }
        }
        return best;
    }

    static int[] twoOpt(double[][] d, int[] start) {
        int[] tour = rotateToZero(start.clone());
        double current = cycleCost(d, tour);
        while (true) {
            int n = tour.length;
            double bestDelta = -EPS;
            int[] bestTour = null;
            for (int a = 0; a < n; a++) {
                for (int b = a + 1; b < n; b++) {
                    if (b == a + 1) continue;
                    int[] candidate = rotateToZero(apply2Opt(tour, a, b));
                    double delta = cycleCost(d, candidate) - current;
                    if (delta < bestDelta) {
                        bestDelta = delta;
                        bestTour = candidate;
                    }
                }
            }
            if (bestTour == null) return tour;
            tour = bestTour;
            current = cycleCost(d, tour);
        }
    }

    static int[] threeOpt(double[][] d, int[] start) {
        int[] tour = rotateToZero(start.clone());
        double current = cycleCost(d, tour);
        while (true) {
            int n = tour.length;
            double bestDelta = -EPS;
            int[] bestTour = null;
            for (int a = 0; a < n - 2; a++) {
                for (int b = a + 1; b < n - 1; b++) {
                    for (int c = b + 1; c < n; c++) {
                        for (int variant = 1; variant <= 7; variant++) {
                            int[] candidate = rotateToZero(apply3Opt(tour, a, b, c, variant));
                            double delta = cycleCost(d, candidate) - current;
                            if (delta < bestDelta) {
                                bestDelta = delta;
                                bestTour = candidate;
                            }
                        }
                    }
                }
            }
            if (bestTour == null) return tour;
            tour = bestTour;
            current = cycleCost(d, tour);
        }
    }

    static boolean improvingThreeOpt(double[][] d, int[] tour) {
        return minThreeOptDelta(d, tour) < -EPS;
    }

    static boolean isPermutation(int[] tour, int n) {
        boolean[] seen = new boolean[n];
        for (int value : tour) {
            if (value < 0 || value >= n || seen[value]) return false;
            seen[value] = true;
        }
        return true;
    }

    static double optimum(double[][] d) {
        int n = d.length;
        int[] rest = new int[n - 1];
        for (int i = 0; i < n - 1; i++) rest[i] = i + 1;
        double best = Double.POSITIVE_INFINITY;
        while (true) {
            int[] tour = new int[n];
            tour[0] = 0;
            System.arraycopy(rest, 0, tour, 1, n - 1);
            if (tour[1] < tour[n - 1]) best = Math.min(best, cycleCost(d, tour));
            int i = n - 3;
            while (i >= 0 && rest[i] > rest[i + 1]) i--;
            if (i < 0) break;
            int j = n - 2;
            while (rest[j] < rest[i]) j--;
            int tmp = rest[i];
            rest[i] = rest[j];
            rest[j] = tmp;
            for (int lo = i + 1, hi = n - 2; lo < hi; lo++, hi--) {
                tmp = rest[lo];
                rest[lo] = rest[hi];
                rest[hi] = tmp;
            }
        }
        return best;
    }

    static double[][] circle(int n, double radius) {
        double[][] pts = new double[n][2];
        for (int i = 0; i < n; i++) {
            double angle = 2.0 * Math.PI * i / n;
            pts[i][0] = radius * Math.cos(angle);
            pts[i][1] = radius * Math.sin(angle);
        }
        return pts;
    }

    static double[][] lcgPoints(int n, int seed, int limit) {
        double[][] pts = new double[n][2];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            pts[i][0] = seed % limit;
            seed = (seed * 75 + 74) % 65537;
            pts[i][1] = seed % limit;
        }
        return pts;
    }

    public static void main(String[] args) {
        double[][] squarePts = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
        double[][] square = distances(squarePts);
        for (int i = 0; i < 4; i++) check(Math.abs(square[i][i]) <= EPS, "zero self distance");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                check(Math.abs(square[i][j] - square[j][i]) <= EPS, "distance matrix is symmetric");
            }
        }
        int[] squareNn = nearestNeighbor(square);
        check(Arrays.equals(squareNn, new int[]{0, 1, 2, 3}), "nearest neighbour on the unit square");
        check(Math.abs(cycleCost(square, squareNn) - 4.0) <= EPS, "unit square perimeter is 4");
        check(Math.abs(optimum(square) - 4.0) <= EPS, "brute force optimum of the unit square is 4");

        int n = 10;
        double[][] convex = distances(circle(n, 1000.0));
        double hullPerimeter = 0.0;
        for (int i = 0; i < n; i++) hullPerimeter += convex[i][(i + 1) % n];
        int[] convex2opt = twoOpt(convex, nearestNeighbor(convex));
        check(isPermutation(convex2opt, n), "2-opt tour is a permutation");
        check(Math.abs(cycleCost(convex, convex2opt) - hullPerimeter) <= 1e-6,
                "points in convex position: the 2-optimal tour is optimal");
        check(minTwoOptDelta(convex, convex2opt) >= -EPS, "convex 2-opt tour has no improving move");
        int[] convex3opt = threeOpt(convex, nearestNeighbor(convex));
        check(Math.abs(cycleCost(convex, convex3opt) - hullPerimeter) <= 1e-6,
                "3-opt also reaches the convex hull perimeter");
        check(!improvingThreeOpt(convex, convex3opt), "convex 3-opt tour has no improving move");

        int[] seeds = {17, 29, 41, 53, 65, 77, 89, 101, 113, 125, 137, 149};
        int strictImprovements = 0;
        int threeOptBeatsTwoOpt = 0;
        int instances = 0;
        for (int s = 0; s < seeds.length; s++) {
            int points = 8;
            double[][] d = distances(lcgPoints(points, seeds[s], 60));
            int[] nn = nearestNeighbor(d);
            int[] two = twoOpt(d, nn);
            int[] three = threeOpt(d, nn);
            double nnCost = cycleCost(d, nn);
            double twoCost = cycleCost(d, two);
            double threeCost = cycleCost(d, three);
            double best = optimum(d);
            instances++;
            check(isPermutation(two, points) && isPermutation(three, points), "tour validity");
            check(twoCost <= nnCost + EPS, "2-opt never loses to nearest neighbour");
            check(threeCost <= twoCost + EPS, "3-opt never loses to 2-opt");
            check(twoCost >= best - EPS && threeCost >= best - EPS, "no heuristic beats the optimum");
            check(threeCost <= 1.5 * best + EPS, "3-opt stays within a documented factor of optimal");
            check(cycleCost(d, twoOpt(d, two)) == twoCost, "2-opt is idempotent");
            check(cycleCost(d, threeOpt(d, three)) == threeCost, "3-opt is idempotent");
            if (twoCost > best + EPS) strictImprovements++;
            double fromTwo = cycleCost(d, threeOpt(d, two));
            check(fromTwo <= twoCost + EPS, "3-opt never increases the cost of a 2-opt tour");
            check((fromTwo < twoCost - EPS) == improvingThreeOpt(d, two),
                    "an improving 3-opt move exists iff the search strictly improves");
            if (fromTwo < twoCost - EPS) threeOptBeatsTwoOpt++;
        }
        check(instances == seeds.length, "all instances evaluated");
        check(strictImprovements >= 1, "at least one instance shows 2-opt is not optimal");
        check(threeOptBeatsTwoOpt >= 1, "at least one instance where 3-opt strictly beats 2-opt");

        double[][] line = distances(new double[][]{{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}});
        int[] lineTour = twoOpt(line, nearestNeighbor(line));
        check(Arrays.equals(lineTour, new int[]{0, 1, 2, 3, 4}), "collinear points keep their order");
        check(Math.abs(cycleCost(line, lineTour) - 8.0) <= EPS, "line tour traverses the segment twice");
        check(Arrays.equals(nearestNeighbor(distances(new double[][]{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}})),
                new int[]{0, 1, 2}), "identical points fall back to the smallest index");

        int[] single = twoOpt(distances(new double[][]{{5.0, 5.0}}), new int[]{0});
        check(single.length == 1 && Math.abs(cycleCost(distances(new double[][]{{5.0, 5.0}}), single)) <= EPS,
                "single point tour costs 0");
        int[] pair = twoOpt(distances(new double[][]{{0.0, 0.0}, {3.0, 4.0}}), new int[]{0, 1});
        check(Math.abs(cycleCost(distances(new double[][]{{0.0, 0.0}, {3.0, 4.0}}), pair) - 10.0) <= EPS,
                "two-point tour goes there and back");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
