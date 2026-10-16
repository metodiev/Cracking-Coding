/** Task 557 - Implement algebraic path finding with min-plus matrix multiply (tropical semiring). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Algebraic path finding over the tropical (min-plus) semiring: addition is minimum, multiplication is
 * ordinary addition, the additive identity is +infinity and the multiplicative identity is 0, so
 * (W^k)[i][j] is the lightest walk from i to j using exactly k edges and the Kleene star
 * W* = min over k = 1..n-1 of W^k is the all-pairs shortest path matrix. The self-test implements the
 * semiring product by the ordinary O(n^3) loop with explicit +infinity handling (an infinite operand is
 * skipped so that no saturated addition can overflow), powers by binary exponentiation, the closure by
 * accumulation of successive powers, and standard Floyd–Warshall as an independent route.
 *
 * Documented checks. Semiring laws are verified numerically on seeded random matrices: add is
 * associative, commutative and idempotent with +infinity as its identity; multiplication is associative
 * with the tropical identity matrix; multiplication distributes over addition on both sides; and
 * A^m (x) A^p = A^(m+p) for m, p <= 4, which is the associativity identity the closure argument rests
 * on. The structural fact that min-plus is *not* a ring — it has no additive inverses, so cancellation
 * fails — is asserted with the witness min(1,3) = min(1,5) = 1 while 3 != 5. Correctness of the powers:
 * for k = 1..5 the min-plus power equals the optimum over an explicit enumeration of all walks of
 * exactly k edges, entrywise, and W^0 is the tropical identity (the zero-length stay). Shortest paths:
 * the closure equals Floyd–Warshall, equals a per-source Dijkstra run on the positive instance, and
 * equals the lightest simple path found by brute-force DFS — off the diagonal, since the diagonal of the
 * closure is the lightest closed walk, which the simple-path enumeration deliberately cannot see
 * (checked instead against the optimality condition closure[i][i] = min_j w[i][j] + closure[j][i]).
 * Negative weights are exercised on acyclic instances (edges only from i to j > i, 60% positive and 40%
 * negative), where the closure, Floyd–Warshall and the lightest simple path all still agree and the
 * diagonal stays infinite.
 *
 * The honest caveat is asserted rather than hidden: on a graph with a negative cycle the closure stops
 * being a shortest-path matrix because min over k <= n-1 edges keeps decreasing as k grows. The
 * self-test builds a 3-cycle of total weight -1 (plus a pendant edge), asserts that the 3-edge power
 * gives -1 on its diagonal, that the 6-edge and 9-edge powers give strictly smaller values, and that
 * the closure diagonal is negative — which is exactly how a negative cycle is certified — while the
 * lightest simple closed walk does not exist at all (the brute force reports +infinity). Boundaries
 * covered: the empty graph, a single vertex with and without a self-loop, the edgeless all-infinity
 * matrix, infinity as an absorbing element (infinity + infinity must stay infinity rather than
 * overflow), zero-weight edges, and a deterministic re-run.
 */
import java.util.*;

public class Solution {

    static final long INF = Long.MAX_VALUE / 4;

    static final class Rng {
        private long state;

        Rng(long seed) {
            this.state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        double nextDouble() {
            return (next() >>> 11) * 0x1.0p-53;
        }

        int nextInt(int bound) {
            return (int) ((next() >>> 1) % bound);
        }
    }

    /**
     * Random weighted digraph. Positive instance: every off-diagonal pair carries 1 + rnd(9).
     * Negative instance: a DAG on 0..n-1 (edges only from i to j > i) whose edges are 1 + rnd(9) with
     * probability 0.6 and -(1 + rnd(4)) otherwise, so negative weights are present but no cycle exists —
     * which is exactly the hypothesis under which shortest walks and shortest simple paths coincide.
     */
    static long[][] randomWeights(int n, boolean allowNegative, Rng rng) {
        long[][] w = new long[n][n];
        for (long[] row : w) Arrays.fill(row, INF);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                if (!allowNegative) {
                    w[i][j] = 1 + rng.nextInt(9);
                } else if (j > i) {
                    w[i][j] = rng.nextDouble() < 0.6 ? 1 + rng.nextInt(9) : -(1 + rng.nextInt(4));
                }
            }
        }
        return w;
    }

    static long tropicalAdd(long a, long b) {
        return Math.min(a, b);
    }

    static long[][] tropicalAddMatrix(long[][] a, long[][] b) {
        int n = a.length;
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) out[i][j] = tropicalAdd(a[i][j], b[i][j]);
        }
        return out;
    }

    static long[][] tropicalMultiply(long[][] a, long[][] b) {
        int n = a.length;
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                long best = INF;
                for (int k = 0; k < n; k++) {
                    if (a[i][k] == INF || b[k][j] == INF) continue;
                    long candidate = a[i][k] + b[k][j];
                    if (candidate < best) best = candidate;
                }
                out[i][j] = best;
            }
        }
        return out;
    }

    static long[][] tropicalIdentity(int n) {
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) out[i][j] = i == j ? 0 : INF;
        }
        return out;
    }

    static long[][] tropicalPower(long[][] base, int exponent) {
        int n = base.length;
        long[][] result = tropicalIdentity(n);
        long[][] square = new long[n][n];
        for (int i = 0; i < n; i++) square[i] = base[i].clone();
        int e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) result = tropicalMultiply(result, square);
            square = tropicalMultiply(square, square);
            e >>= 1;
        }
        return result;
    }

    /** Kleene star by accumulating min over k = 1..n-1 of W^k. */
    static long[][] tropicalClosureByPower(long[][] w) {
        int n = w.length;
        if (n == 0) return new long[0][0];
        long[][] current = w;
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) out[i] = w[i].clone();
        for (int step = 2; step <= n - 1; step++) {
            current = tropicalMultiply(current, w);
            out = tropicalAddMatrix(out, current);
        }
        return out;
    }

    /** Floyd–Warshall in the tropical semiring. */
    static long[][] tropicalClosureFloyd(long[][] w) {
        int n = w.length;
        long[][] d = new long[n][n];
        for (int i = 0; i < n; i++) d[i] = w[i].clone();
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                if (d[i][k] == INF) continue;
                for (int j = 0; j < n; j++) {
                    if (d[k][j] == INF) continue;
                    long candidate = d[i][k] + d[k][j];
                    if (candidate < d[i][j]) d[i][j] = candidate;
                }
            }
        }
        return d;
    }

    /** Dijkstra with a simple O(n^2) selection scan; valid for non-negative weights. */
    static long[] dijkstra(long[][] w, int source) {
        int n = w.length;
        long[] dist = new long[n];
        boolean[] done = new boolean[n];
        Arrays.fill(dist, INF);
        dist[source] = 0;
        for (int step = 0; step < n; step++) {
            int best = -1;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist[v] < INF && (best < 0 || dist[v] < dist[best])) best = v;
            }
            if (best < 0) break;
            done[best] = true;
            for (int v = 0; v < n; v++) {
                if (w[best][v] == INF || done[v]) continue;
                long candidate = dist[best] + w[best][v];
                if (candidate < dist[v]) dist[v] = candidate;
            }
        }
        return dist;
    }

    /** Lightest walk of exactly k edges, by enumerating every walk. */
    static long[][] bruteWalkOptimum(long[][] w, int k) {
        int n = w.length;
        long[][] out = new long[n][n];
        for (int start = 0; start < n; start++) {
            long[] best = new long[n];
            Arrays.fill(best, INF);
            bruteWalk(w, start, start, k, 0, best);
            out[start] = best;
        }
        return out;
    }

    private static void bruteWalk(long[][] w, int start, int v, int remaining, long cost, long[] best) {
        if (remaining == 0) {
            if (cost < best[v]) best[v] = cost;
            return;
        }
        for (int j = 0; j < w.length; j++) {
            if (w[v][j] == INF) continue;
            bruteWalk(w, start, j, remaining - 1, cost + w[v][j], best);
        }
    }

    /** Lightest simple path between every pair; valid even with negative edges. */
    static long[][] bruteSimplePaths(long[][] w) {
        int n = w.length;
        long[][] out = new long[n][n];
        for (int start = 0; start < n; start++) {
            long[] best = new long[n];
            Arrays.fill(best, INF);
            boolean[] visited = new boolean[n];
            visited[start] = true;
            best[start] = Math.min(best[start], w[start][start]);
            simpleDfs(w, start, start, 0, visited, best);
            out[start] = best;
        }
        return out;
    }

    private static void simpleDfs(long[][] w, int start, int v, long cost, boolean[] visited,
                                  long[] best) {
        for (int j = 0; j < w.length; j++) {
            if (w[v][j] == INF || visited[j]) continue;
            long candidate = cost + w[v][j];
            if (candidate < best[j]) best[j] = candidate;
            visited[j] = true;
            simpleDfs(w, start, j, candidate, visited, best);
            visited[j] = false;
        }
    }

    static boolean equal(long[][] a, long[][] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) if (a[i][j] != b[i][j]) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        // --- Semiring laws.
        Rng rng = new Rng(20240557L);
        for (int t = 0; t < 5; t++) {
            int n = 3 + t;
            long[][] a = randomWeights(n, t % 2 == 1, rng);
            long[][] b = randomWeights(n, false, rng);
            long[][] c = randomWeights(n, false, rng);
            long[][] id = tropicalIdentity(n);
            check(equal(tropicalMultiply(tropicalMultiply(a, b), c),
                    tropicalMultiply(a, tropicalMultiply(b, c))), "product is associative");
            check(equal(tropicalMultiply(a, id), a) && equal(tropicalMultiply(id, a), a),
                    "the tropical identity is the multiplicative identity");
            check(equal(tropicalAddMatrix(a, b), tropicalAddMatrix(b, a)),
                    "addition is commutative");
            check(equal(tropicalAddMatrix(tropicalAddMatrix(a, b), c),
                    tropicalAddMatrix(a, tropicalAddMatrix(b, c))), "addition is associative");
            check(equal(tropicalAddMatrix(a, a), a), "addition is idempotent");
            long[][] inf = new long[n][n];
            for (long[] row : inf) Arrays.fill(row, INF);
            check(equal(tropicalAddMatrix(a, inf), a), "infinity is the additive identity");
            check(equal(tropicalMultiply(a, tropicalAddMatrix(b, c)),
                    tropicalAddMatrix(tropicalMultiply(a, b), tropicalMultiply(a, c))),
                    "multiplication distributes over addition on the right factor");
            check(equal(tropicalMultiply(tropicalAddMatrix(a, b), c),
                    tropicalAddMatrix(tropicalMultiply(a, c), tropicalMultiply(b, c))),
                    "multiplication distributes over addition on the left factor");
            for (int m = 1; m <= 4; m++) {
                for (int p = 1; p <= 4; p++) {
                    check(equal(tropicalMultiply(tropicalPower(a, m), tropicalPower(a, p)),
                            tropicalPower(a, m + p)), "A^m (x) A^p = A^(m+p) for m = " + m + ", p = " + p);
                }
            }
        }
        check(tropicalAdd(1, 3) == tropicalAdd(1, 5) && 3 != 5,
                "min-plus is not cancellative: min(1,3) = min(1,5) = 1 while 3 != 5");

        // --- Powers agree with explicit walk optima.
        long[][] small = randomWeights(4, true, new Rng(557002L));
        for (int k = 1; k <= 5; k++) {
            check(equal(tropicalPower(small, k), bruteWalkOptimum(small, k)),
                    "W^" + k + " equals the optimum over all walks of exactly " + k + " edges");
        }
        check(equal(tropicalPower(small, 0), tropicalIdentity(4)),
                "W^0 is the tropical identity, i.e. the zero-length stay");

        // --- Shortest paths: closure by accumulation, Floyd–Warshall, Dijkstra and simple-path brute force.
        for (int t = 0; t < 6; t++) {
            int n = 4 + (t % 3);
            long[][] w = randomWeights(n, false, rng);
            long[][] byPower = tropicalClosureByPower(w);
            long[][] byFloyd = tropicalClosureFloyd(w);
            check(equal(byPower, byFloyd), "closure by power accumulation equals Floyd–Warshall");
            long[][] simple = bruteSimplePaths(w);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) check(byPower[i][j] == simple[i][j],
                            "closure equals the lightest simple path " + i + " -> " + j);
                }
                long bestCycle = INF;
                for (int j = 0; j < n; j++) {
                    if (j == i || w[i][j] == INF || byFloyd[j][i] == INF) continue;
                    bestCycle = Math.min(bestCycle, w[i][j] + byFloyd[j][i]);
                }
                check(byPower[i][i] == bestCycle, "the diagonal is the lightest closed walk through "
                        + i + " (" + bestCycle + "), which the simple-path enumeration cannot see");
            }
            for (int source = 0; source < n; source++) {
                long[] dist = dijkstra(w, source);
                for (int target = 0; target < n; target++) {
                    long expected = source == target ? 0 : byFloyd[source][target];
                    check(dist[target] == expected, "Dijkstra from " + source + " to " + target
                            + " agrees with the closure (" + expected + ")");
                }
            }
        }
        for (int t = 0; t < 4; t++) {
            int n = 4 + t;
            long[][] w = randomWeights(n, true, rng);
            long[][] byPower = tropicalClosureByPower(w);
            check(equal(byPower, tropicalClosureFloyd(w)),
                    "on an acyclic instance with negative edges the power closure still equals "
                            + "Floyd–Warshall");
            long[][] simple = bruteSimplePaths(w);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) check(byPower[i][j] == simple[i][j], "with negative edges the closure "
                            + "still equals the lightest simple path " + i + " -> " + j);
                }
            }
        }

        // --- Negative cycle: the closure stops being a shortest-path matrix.
        long[][] negativeCycle = new long[4][4];
        for (long[] row : negativeCycle) Arrays.fill(row, INF);
        negativeCycle[0][1] = -1;
        negativeCycle[1][2] = -1;
        negativeCycle[2][0] = 1;
        negativeCycle[2][3] = 1;
        long[][] atThree = tropicalPower(negativeCycle, 3);
        long[][] atSix = tropicalPower(negativeCycle, 6);
        check(atThree[0][0] == -1, "the 3-cycle has total weight -1");
        check(atSix[0][0] < atThree[0][0], "repeating the negative cycle keeps lowering the walk weight: "
                + atSix[0][0] + " < " + atThree[0][0]);
        check(tropicalClosureByPower(negativeCycle)[0][0] < 0,
                "so the closure diagonal is negative, which certifies the negative cycle");
        check(tropicalPower(negativeCycle, 9)[0][0] < tropicalPower(negativeCycle, 6)[0][0],
                "and further laps keep lowering it, so no finite minimum exists");
        check(bruteSimplePaths(negativeCycle)[0][0] > tropicalClosureByPower(negativeCycle)[0][0],
                "and the simple-path optimum is larger, so the closure is not a shortest-path matrix here");
        long[][] acyclic = new long[3][3];
        for (long[] row : acyclic) Arrays.fill(row, INF);
        acyclic[0][1] = 2;
        acyclic[1][2] = -3;
        check(tropicalClosureByPower(acyclic)[0][2] == -1, "negative edges without a cycle are fine");
        check(tropicalClosureByPower(acyclic)[0][0] == INF, "with no cycle the diagonal stays infinity");

        // --- Boundaries.
        check(tropicalClosureByPower(new long[0][0]).length == 0, "the empty graph");
        long[][] single = {{INF}};
        check(tropicalClosureByPower(single)[0][0] == INF, "a single vertex with no stay has no path");
        long[][] singleLoop = {{0}};
        check(tropicalClosureByPower(singleLoop)[0][0] == 0, "a zero-weight self-loop costs 0");
        long[][] allInf = new long[3][3];
        for (long[] row : allInf) Arrays.fill(row, INF);
        check(equal(tropicalClosureByPower(allInf), allInf), "an edgeless graph has an all-infinity closure");
        check(tropicalMultiply(allInf, allInf)[0][0] == INF,
                "infinity is absorbing: infinity + infinity = infinity, never an overflow");

        check(equal(tropicalPower(small, 3), tropicalPower(small, 3)), "deterministic re-run");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
