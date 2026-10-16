/** Task 550 - Implement graph sketches for neighborhood similarity (MinHash). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * MinHash sketches of vertex neighbourhoods: with the affine family h(x) = (a x + b) mod p over the
 * Mersenne prime p = 2^31 - 1 and a != 0, the probability that two sets attain the same minimum is
 * their Jaccard similarity, so a sketch of k independent minima estimates |A n B| / |A u B| by the
 * fraction of matching coordinates, with standard error sqrt(J(1-J)/k). A second, merge-friendly
 * variant keeps the k smallest values of a single hash function (bottom-k). Both sketches are
 * mergeable — the sketch of a union is the coordinatewise minimum of the sketches of the parts, and
 * for bottom-k it is the k smallest values of the merged sorted lists whenever the sketches are full
 * (|S| >= k) — which the self-test verifies exactly rather than statistically.
 *
 * Documented checks: exact Jaccard values on hand-written sets (identical sets 1, disjoint sets 0
 * and J({1,2,3},{2,3,4}) = 0.5); identical sketches for identical sets and an exactly zero estimate
 * for disjoint sets, since the affine map is injective modulo p and disjoint sets therefore can
 * never produce a colliding minimum; mergeability of both sketch families against a directly
 * computed sketch of the union; on a planted three-cluster LCG graph (three clusters of 8 vertices,
 * 70% edge probability inside and 5% across) the worst absolute error over all comparable vertex
 * pairs with k = 512 minima stays within the documented 0.1, the mean absolute error within 0.03 and
 * the mean signed error within 0.02 (unbiasedness); the sketched and exact similarity rankings agree
 * on at least 14 of the top 20 pairs; within-cluster neighbourhoods average more than three times the
 * cross-cluster similarity (0.37 versus 0.044 on this instance, with the extremes deliberately
 * overlapping, so the claim is made about averages and rankings rather than per pair) and all of the
 * top 20 pairs by both the exact and the sketched similarity lie inside a cluster; the sketch costs
 * exactly k * |S| hash evaluations; and a fixed seed reproduces
 * every sketch, while a different seed does not.
 */
import java.util.*;

public class Solution {
    static final long P = 2147483647L;

    /** k affine hash functions (a != 0) generated deterministically from the seed. */
    static long[][] parameters(int k, int seed) {
        long[][] out = new long[k][2];
        for (int i = 0; i < k; i++) {
            seed = (seed * 75 + 74) % 65537;
            long high = seed;
            seed = (seed * 75 + 74) % 65537;
            long low = seed;
            out[i][0] = 1 + (high * 65537 + low) % (P - 1);
            seed = (seed * 75 + 74) % 65537;
            long high2 = seed;
            seed = (seed * 75 + 74) % 65537;
            long low2 = seed;
            out[i][1] = (high2 * 65537 + low2) % P;
        }
        return out;
    }

    static long hash(long x, long a, long b) {
        return (a * x + b) % P;
    }

    /** k minima, one per hash function; the empty set is represented by the sentinel -1. */
    static long[] minHashSketch(int[] members, long[][] params, long[] evaluations) {
        long[] out = new long[params.length];
        for (int i = 0; i < params.length; i++) {
            long best = Long.MAX_VALUE;
            for (int x : members) {
                if (evaluations != null) evaluations[0]++;
                best = Math.min(best, hash(x, params[i][0], params[i][1]));
            }
            out[i] = members.length == 0 ? -1 : best;
        }
        return out;
    }

    static double minHashEstimate(long[] left, long[] right) {
        if (left.length == 0 || left[0] < 0 || right[0] < 0) return 0.0;
        int matches = 0;
        for (int i = 0; i < left.length; i++) {
            if (left[i] == right[i]) matches++;
        }
        return (double) matches / left.length;
    }

    /** Bottom-k sketch: the k smallest hash values under one hash function, ascending. */
    static long[] bottomK(int[] members, long a, long b, int k) {
        long[] values = new long[members.length];
        for (int i = 0; i < members.length; i++) values[i] = hash(members[i], a, b);
        Arrays.sort(values);
        return Arrays.copyOf(values, Math.min(k, values.length));
    }

    static long[] mergeBottomK(long[] left, long[] right, int k) {
        long[] combined = new long[left.length + right.length];
        System.arraycopy(left, 0, combined, 0, left.length);
        System.arraycopy(right, 0, combined, left.length, right.length);
        Arrays.sort(combined);
        return Arrays.copyOf(combined, Math.min(k, combined.length));
    }

    static double bottomKEstimate(long[] left, long[] right) {
        if (left.length == 0 || right.length == 0) return 0.0;
        int i = 0;
        int j = 0;
        int matches = 0;
        while (i < left.length && j < right.length) {
            if (left[i] == right[j]) {
                matches++;
                i++;
                j++;
            } else if (left[i] < right[j]) {
                i++;
            } else {
                j++;
            }
        }
        return (double) matches / Math.min(left.length, right.length);
    }

    static int[] asArray(Collection<Integer> values) {
        int[] out = new int[values.size()];
        int at = 0;
        for (int value : values) out[at++] = value;
        return out;
    }

    static int[] union(int[] a, int[] b) {
        Set<Integer> set = new TreeSet<>();
        for (int value : a) set.add(value);
        for (int value : b) set.add(value);
        return asArray(set);
    }

    static double exactJaccard(int[] a, int[] b) {
        Set<Integer> left = new TreeSet<>();
        Set<Integer> right = new TreeSet<>();
        for (int value : a) left.add(value);
        for (int value : b) right.add(value);
        if (left.isEmpty() && right.isEmpty()) return 0.0;
        int intersection = 0;
        for (int value : left) {
            if (right.contains(value)) intersection++;
        }
        return (double) intersection / (left.size() + right.size() - intersection);
    }

    static int[] neighbourhood(boolean[][] adj, int v, int n) {
        List<Integer> list = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            if (adj[v][u]) list.add(u);
        }
        return asArray(list);
    }

    /** Planted three-cluster graph: dense inside a cluster, sparse across clusters. */
    static boolean[][] plantedGraph(int clusterSize, int inside, int across, int seed) {
        int n = 3 * clusterSize;
        boolean[][] adj = new boolean[n][n];
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                seed = (seed * 75 + 74) % 65537;
                int threshold = (a / clusterSize == b / clusterSize) ? inside : across;
                if (seed % 100 < threshold) {
                    adj[a][b] = true;
                    adj[b][a] = true;
                }
            }
        }
        return adj;
    }

    static List<int[]> allPairs(int n) {
        List<int[]> out = new ArrayList<>();
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) out.add(new int[]{a, b});
        }
        return out;
    }

    static List<int[]> ranked(List<int[]> pairs, double[][] matrix) {
        List<int[]> copy = new ArrayList<>(pairs);
        copy.sort((x, y) -> Double.compare(matrix[y[0]][y[1]], matrix[x[0]][x[1]]));
        return copy;
    }

    public static void main(String[] args) {
        int[] setA = {1, 2, 3};
        int[] setB = {2, 3, 4};
        int[] setC = {7, 8, 9};
        check(Math.abs(exactJaccard(setA, setA) - 1.0) <= 1e-15, "self similarity is 1");
        check(exactJaccard(setA, setC) == 0.0, "disjoint sets have Jaccard 0");
        check(Math.abs(exactJaccard(setA, setB) - 0.5) <= 1e-15, "J({1,2,3},{2,3,4}) = 2/4");
        check(exactJaccard(new int[0], new int[0]) == 0.0, "two empty sets score 0 by convention");
        check(Math.abs(exactJaccard(setA, union(setA, setB)) - 3.0 / 4.0) <= 1e-15, "union similarity");

        long[][] params = parameters(512, 20240101);
        long[] sketchA = minHashSketch(setA, params, null);
        long[] sketchB = minHashSketch(setB, params, null);
        long[] sketchC = minHashSketch(setC, params, null);
        check(sketchA.length == 512, "the sketch has exactly k coordinates");
        check(minHashEstimate(sketchA, sketchA) == 1.0, "identical sets estimate exactly 1");
        check(minHashEstimate(sketchA, sketchC) == 0.0,
                "disjoint sets can never collide, so the estimate is exactly 0");
        check(Math.abs(minHashEstimate(sketchA, sketchB) - 0.5) <= 0.1,
                "the estimate for J = 0.5 is within 0.1 with k = 512");
        check(minHashEstimate(new long[0], sketchA) == 0.0, "an empty sketch scores 0");
        check(minHashEstimate(minHashSketch(new int[0], params, null), minHashSketch(new int[0], params, null)) == 0.0,
                "two empty neighbourhoods score 0");

        long[] evaluations = new long[1];
        minHashSketch(setA, params, evaluations);
        check(evaluations[0] == 512L * setA.length, "the sketch costs exactly k * |S| hash evaluations");
        check(Arrays.equals(minHashSketch(setA, parameters(512, 20240101), null), sketchA),
                "the same seed reproduces the sketch");
        check(!Arrays.equals(minHashSketch(setA, parameters(512, 777), null), sketchA),
                "a different seed gives a different sketch");

        long[] mergeSketch = minHashSketch(union(setA, setB), params, null);
        long[] minOfSketches = new long[512];
        for (int i = 0; i < 512; i++) minOfSketches[i] = Math.min(sketchA[i], sketchB[i]);
        check(Arrays.equals(mergeSketch, minOfSketches),
                "MinHash sketches merge by coordinatewise minimum");

        long[] bottomA = bottomK(setA, 1234567L, 89L, 2);
        long[] bottomB = bottomK(setB, 1234567L, 89L, 2);
        check(Arrays.equals(bottomK(union(setA, setB), 1234567L, 89L, 2), mergeBottomK(bottomA, bottomB, 2)),
                "truncated bottom-k sketches merge by taking the k smallest values");
        int[] bigA = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
        int[] bigB = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
        long[] fullA = bottomK(bigA, 999983L, 7L, 5);
        long[] fullB = bottomK(bigB, 999983L, 7L, 5);
        check(Arrays.equals(bottomK(union(bigA, bigB), 999983L, 7L, 5), mergeBottomK(fullA, fullB, 5)),
                "full bottom-k sketches merge exactly");
        check(bottomK(setA, 1234567L, 89L, 100).length == setA.length,
                "the bottom-k sketch never exceeds the set size");
        check(bottomKEstimate(bottomA, bottomA) == 1.0, "bottom-k self similarity is 1");
        check(bottomKEstimate(bottomA, bottomK(setC, 1234567L, 89L, 8)) == 0.0,
                "bottom-k gives 0 for disjoint sets");

        int cluster = 8;
        int n = 3 * cluster;
        boolean[][] plant = plantedGraph(cluster, 70, 5, 424242);
        List<int[]> pairs = allPairs(n);
        long[][] sketchOf = new long[n][];
        long[] calls = new long[1];
        for (int v = 0; v < n; v++) {
            sketchOf[v] = minHashSketch(neighbourhood(plant, v, n), params, calls);
        }
        double[][] exactMatrix = new double[n][n];
        double[][] sketchMatrix = new double[n][n];
        double worst = 0.0;
        double meanAbsolute = 0.0;
        double meanSigned = 0.0;
        int comparable = 0;
        for (int[] pair : pairs) {
            int[] left = neighbourhood(plant, pair[0], n);
            int[] right = neighbourhood(plant, pair[1], n);
            if (left.length == 0 || right.length == 0) continue;
            double exact = exactJaccard(left, right);
            double estimate = minHashEstimate(sketchOf[pair[0]], sketchOf[pair[1]]);
            exactMatrix[pair[0]][pair[1]] = exact;
            exactMatrix[pair[1]][pair[0]] = exact;
            sketchMatrix[pair[0]][pair[1]] = estimate;
            sketchMatrix[pair[1]][pair[0]] = estimate;
            worst = Math.max(worst, Math.abs(estimate - exact));
            meanAbsolute += Math.abs(estimate - exact);
            meanSigned += estimate - exact;
            comparable++;
            check(estimate >= 0.0 && estimate <= 1.0, "estimates stay inside [0, 1]");
        }
        meanAbsolute /= comparable;
        meanSigned /= comparable;
        check(comparable > 200, "the planted graph yields plenty of comparable pairs");
        check(calls[0] == 512L * totalDegree(plant, n), "the whole sketch bank costs k * sum of degrees");
        check(worst <= 0.1, "worst absolute error " + worst + " stays within 0.1");
        check(meanAbsolute <= 0.03, "mean absolute error " + meanAbsolute + " stays within 0.03");
        check(Math.abs(meanSigned) <= 0.02, "the estimator is unbiased to within 0.02");

        List<int[]> byExact = ranked(pairs, exactMatrix);
        List<int[]> bySketch = ranked(pairs, sketchMatrix);
        int overlap = 0;
        for (int i = 0; i < 20; i++) {
            int[] top = byExact.get(i);
            for (int j = 0; j < 20; j++) {
                int[] candidate = bySketch.get(j);
                if (top[0] == candidate[0] && top[1] == candidate[1]) {
                    overlap++;
                    break;
                }
            }
        }
        check(overlap >= 14, "the sketched ranking matches the exact one on " + overlap + "/20 top pairs");

        double withinMin = Double.POSITIVE_INFINITY;
        double acrossMax = Double.NEGATIVE_INFINITY;
        double withinMean = 0.0;
        double acrossMean = 0.0;
        int withinCount = 0;
        int acrossCount = 0;
        for (int[] pair : pairs) {
            int[] left = neighbourhood(plant, pair[0], n);
            int[] right = neighbourhood(plant, pair[1], n);
            if (left.length == 0 || right.length == 0) continue;
            double estimate = sketchMatrix[pair[0]][pair[1]];
            if (pair[0] / cluster == pair[1] / cluster) {
                withinMin = Math.min(withinMin, estimate);
                withinMean += estimate;
                withinCount++;
            } else {
                acrossMax = Math.max(acrossMax, estimate);
                acrossMean += estimate;
                acrossCount++;
            }
        }
        withinMean /= withinCount;
        acrossMean /= acrossCount;
        check(withinCount > 0 && acrossCount > 0, "both within and across cluster pairs exist");
        int topWithin = 0;
        for (int i = 0; i < 20; i++) {
            int[] top = byExact.get(i);
            if (top[0] / cluster == top[1] / cluster) topWithin++;
        }
        int topWithinSketch = 0;
        for (int i = 0; i < 20; i++) {
            int[] top = bySketch.get(i);
            if (top[0] / cluster == top[1] / cluster) topWithinSketch++;
        }
        check(withinMean > 3.0 * acrossMean,
                "within-cluster neighbourhoods average " + withinMean + " versus " + acrossMean + " across clusters");
        check(withinMin < acrossMax,
                "the extremes overlap (" + withinMin + " vs " + acrossMax
                        + "), so the check is on averages and rankings, not per-pair dominance");
        check(topWithin == 20 && topWithinSketch == 20,
                "both the exact and the sketched ranking put 20 of the top 20 pairs inside a cluster");

        double self = 0.0;
        for (int v = 0; v < n; v++) {
            self = Math.max(self, Math.abs(minHashEstimate(sketchOf[v], sketchOf[v]) - 1.0));
        }
        check(self <= 1e-15, "each vertex is exactly similar to itself");

        System.out.println("PASS");
    }

    static long totalDegree(boolean[][] adj, int n) {
        long total = 0;
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (adj[u][v]) total++;
            }
        }
        return total;
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
