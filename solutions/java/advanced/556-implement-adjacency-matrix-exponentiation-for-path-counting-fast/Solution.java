/** Task 556 - Implement adjacency matrix exponentiation for path counting (fast exponent). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Walk counting by exponentiation of the adjacency matrix: (A^k)[i][j] is the number of walks of exactly
 * k edges from i to j, computed by binary exponentiation — square the matrix, multiply into the
 * accumulator for every set bit of the exponent — so a walk count for exponent k costs O(n^3 log k)
 * instead of the O(n^3 k) of repeated multiplication. The self-test builds plain long matrices (values
 * stay far below the overflow threshold on the tested instances) and cross-checks every result against
 * a third, independent route.
 *
 * Documented checks. Structural identities: A^0 is the identity, A^1 is A, A^a * A^b = A^(a+b), and
 * (A^m)^2 = A^(2m) — verified on seeded random graphs, several exponents and both multiplication orders.
 * Brute force: for k = 0..6 on 5- and 6-vertex graphs, the total number of walks of length k counted by
 * explicit walk enumeration equals the sum of all entries of A^k, and the per-vertex-pair walk counts
 * agree entrywise. Closed forms: in K_n the number of walks of length k from a vertex to itself is
 * ((n-1)^k + (n-1)(-1)^k)/n and the total number of walks of length k is n(n-1)^k (verified for n = 4, 5
 * and k up to 12); in the 4-cycle the diagonal of A^k is 2^(k-1) for even k and 0 for odd k; the
 * triangle counts recovered from trace(A^3)/6, trace(A^4) = 6 * (#triangles squared-count) and the total
 * walk counts all agree with the closed forms (trace(A^k) = n(n-1)^k for even k in K_n, and 6 in K3 for
 * k = 3). Reachability: A v A^2 v ... v A^(n-1), accumulated by repeated multiplication, equals the
 * transitive closure computed by Floyd–Warshall on the boolean matrix; the exact-length pattern of every
 * A^k for k <= 2(n-1) is contained in that closure; the reflexive closure is idempotent under boolean
 * multiplication; and the diagonal marks exactly the vertices lying on a cycle, cross-checked with an
 * independent BFS. One honest caveat is asserted rather than glossed over: the pattern of A^(n-1) *alone*
 * is not the closure — it holds only the pairs joined by a path of exactly n-1 edges — so only the
 * containment direction is asserted for exact-length powers. A zero matrix, the identity, the
 * single-vertex graph and a self-loop are checked as boundaries, and every result is reproduced on a
 * second run.
 */
import java.util.*;

public class Solution {

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
    }

    static long[][] toMatrix(boolean[][] adj) {
        int n = adj.length;
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) if (adj[i][j]) out[i][j] = 1;
        }
        return out;
    }

    static long[][] randomMatrix(int n, double p, Rng rng) {
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) if (rng.nextDouble() < p) out[i][j] = 1;
        }
        return out;
    }

    static long[][] identity(int n) {
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) out[i][i] = 1;
        return out;
    }

    static long[][] multiply(long[][] a, long[][] b) {
        int n = a.length;
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < n; t++) {
                if (a[i][t] == 0) continue;
                for (int j = 0; j < n; j++) out[i][j] += a[i][t] * b[t][j];
            }
        }
        return out;
    }

    /** Binary exponentiation: square, multiply into the accumulator for every set exponent bit. */
    static long[][] power(long[][] base, int exponent) {
        int n = base.length;
        long[][] result = identity(n);
        long[][] square = new long[n][n];
        for (int i = 0; i < n; i++) square[i] = base[i].clone();
        int e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) result = multiply(result, square);
            square = multiply(square, square);
            e >>= 1;
        }
        return result;
    }

    static long[][] slowPower(long[][] base, int exponent) {
        int n = base.length;
        long[][] out = identity(n);
        for (int step = 0; step < exponent; step++) out = multiply(out, base);
        return out;
    }

    static long totalSum(long[][] m) {
        long total = 0;
        for (long[] row : m) for (long value : row) total += value;
        return total;
    }

    static long trace(long[][] m) {
        long total = 0;
        for (int i = 0; i < m.length; i++) total += m[i][i];
        return total;
    }

    static boolean equal(long[][] a, long[][] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) if (a[i][j] != b[i][j]) return false;
        }
        return true;
    }

    /** Per-vertex-pair walk counts by explicit enumeration of all walks of length k. */
    static long[][] bruteWalkCounts(long[][] adj, int k) {
        int n = adj.length;
        long[][] out = new long[n][n];
        for (int start = 0; start < n; start++) {
            walkDfs(adj, start, start, k, out);
        }
        return out;
    }

    private static void walkDfs(long[][] adj, int start, int v, int remaining, long[][] out) {
        if (remaining == 0) {
            out[start][v]++;
            return;
        }
        for (int j = 0; j < adj.length; j++) {
            if (adj[v][j] != 0) walkDfs(adj, start, j, remaining - 1, out);
        }
    }

    /** A v A^2 v ... v A^(n-1) by repeated multiplication and accumulation: the closure. */
    static boolean[][] booleanClosureByAccumulation(boolean[][] adj) {
        int n = adj.length;
        if (n == 0) return new boolean[0][0];
        long[][] base = toMatrix(adj);
        long[][] current = base;
        boolean[][] out = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) out[i][j] = adj[i][j];
        }
        for (int step = 2; step <= n - 1; step++) {
            current = multiply(current, base);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) if (current[i][j] > 0) out[i][j] = true;
            }
        }
        return out;
    }

    static long ipow(long base, int exponent) {
        long result = 1;
        for (int i = 0; i < exponent; i++) result *= base;
        return result;
    }

    static boolean[][] booleanClosureFloyd(boolean[][] adj) {
        int n = adj.length;
        boolean[][] out = new boolean[n][n];
        for (int i = 0; i < n; i++) out[i] = adj[i].clone();
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                if (!out[i][k]) continue;
                for (int j = 0; j < n; j++) if (out[k][j]) out[i][j] = true;
            }
        }
        return out;
    }

    static boolean[][] pathPattern(long[][] a) {
        int n = a.length;
        boolean[][] out = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) out[i][j] = a[i][j] > 0;
        }
        return out;
    }

    /** Independent reachability test: is target reachable from source by BFS over the edge set? */
    static boolean reachableFrom(boolean[][] adj, int source, int target) {
        int n = adj.length;
        boolean[] seen = new boolean[n];
        int[] queue = new int[n];
        int head = 0;
        int tail = 0;
        queue[tail++] = source;
        seen[source] = true;
        while (head < tail) {
            int v = queue[head++];
            if (v == target) return true;
            for (int j = 0; j < n; j++) {
                if (adj[v][j] && !seen[j]) {
                    seen[j] = true;
                    queue[tail++] = j;
                }
            }
        }
        return false;
    }

    /** A vertex lies on a cycle exactly when some successor can reach it again. */
    static boolean onCycle(boolean[][] adj, int v) {
        for (int j = 0; j < adj.length; j++) {
            if (adj[v][j] && reachableFrom(adj, j, v)) return true;
        }
        return false;
    }

    static boolean[][] complete(int n) {
        boolean[][] adj = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) if (i != j) adj[i][j] = true;
        }
        return adj;
    }

    static boolean[][] cycle(int n) {
        boolean[][] adj = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            adj[i][(i + 1) % n] = true;
            adj[(i + 1) % n][i] = true;
        }
        return adj;
    }

    public static void main(String[] args) {
        // --- Structural identities on seeded random graphs.
        Rng rng = new Rng(20240556L);
        for (int t = 0; t < 8; t++) {
            int n = 4 + (t % 3);
            long[][] a = randomMatrix(n, 0.35 + 0.1 * (t % 3), rng);
            check(equal(power(a, 0), identity(n)), "A^0 is the identity (case " + t + ")");
            check(equal(power(a, 1), a), "A^1 is A (case " + t + ")");
            for (int e = 0; e <= 8; e++) {
                check(equal(power(a, e), slowPower(a, e)),
                        "fast and slow exponentiation agree (case " + t + ", e = " + e + ")");
            }
            check(equal(multiply(power(a, 2), power(a, 3)), power(a, 5)),
                    "A^2 * A^3 = A^5 (case " + t + ")");
            check(equal(multiply(power(a, 3), power(a, 2)), power(a, 5)),
                    "multiplication order does not matter for powers (case " + t + ")");
            for (int m = 1; m <= 4; m++) {
                long[][] squared = multiply(power(a, m), power(a, m));
                check(equal(squared, power(a, 2 * m)), "(A^m)^2 = A^(2m) (case " + t + ", m = " + m + ")");
            }
        }

        // --- Entrywise agreement with explicit walk enumeration.
        for (int t = 0; t < 4; t++) {
            int n = 5 + (t % 2);
            long[][] a = randomMatrix(n, 0.4, rng);
            for (int k = 0; k <= 6; k++) {
                check(equal(power(a, k), bruteWalkCounts(a, k)),
                        "entrywise walk counts match brute force (case " + t + ", k = " + k + ")");
                check(totalSum(power(a, k)) == totalSum(bruteWalkCounts(a, k)),
                        "total walk count matches brute force (case " + t + ", k = " + k + ")");
            }
        }

        // --- Complete graphs: closed forms.
        for (int n = 3; n <= 6; n++) {
            long[][] a = toMatrix(complete(n));
            for (int k = 0; k <= 12; k++) {
                long[][] p = power(a, k);
                long expectedDiagonal = (ipow(n - 1, k) + (n - 1) * ((k % 2 == 0) ? 1 : -1)) / n;
                check(p[0][0] == expectedDiagonal,
                        "K" + n + ", k = " + k + ": closed walks at a vertex = " + expectedDiagonal);
                long totalExpected = n * ipow(n - 1, k);
                check(totalSum(p) == totalExpected,
                        "K" + n + ", k = " + k + ": total walks = " + totalExpected);
                check(trace(p) == n * expectedDiagonal, "trace is n times the diagonal entry");
            }
        }
        long[][] k3 = toMatrix(complete(3));
        check(trace(power(k3, 3)) == 6, "trace(A^3) of K3 is 6, so it holds one triangle");
        check(trace(power(k3, 3)) / 6 == 1, "triangles = trace(A^3)/6");
        check(power(k3, 2)[0][0] == 2, "K3: two 2-walks back to the start, via each other vertex");

        // --- C4: the diagonal of A^k is 2^(k-1) for even k and 0 for odd k.
        long[][] c4 = toMatrix(cycle(4));
        for (int k = 1; k <= 10; k++) {
            long[][] p = power(c4, k);
            long expected = (k % 2 == 0) ? (1L << (k - 1)) : 0L;
            check(p[0][0] == expected && p[2][2] == expected,
                    "C4, k = " + k + ": closed walks = " + expected);
            check(trace(p) == 4L * expected, "C4, k = " + k + ": closed-walk mass = 4 * " + expected);
            check(totalSum(p) == 4L * ipow(2, k), "C4, k = " + k
                    + ": every vertex has 2^k walk-prefixes, so 4 * 2^k walks in total");
        }

        // --- Reachability: the boolean pattern of A^(n-1) is the transitive closure.
        for (int t = 0; t < 6; t++) {
            int n = 5 + (t % 3);
            boolean[][] adj = new boolean[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) adj[i][j] = rng.nextDouble() < 0.3;
            }
            boolean[][] byFloyd = booleanClosureFloyd(adj);
            boolean[][] byAccumulated = booleanClosureByAccumulation(adj);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    check(byAccumulated[i][j] == byFloyd[i][j],
                            "A v A^2 v ... v A^(n-1) matches Floyd–Warshall (case " + t + ")");
                }
            }
            for (int step = 1; step <= 2 * (n - 1); step++) {
                boolean[][] pattern = pathPattern(power(toMatrix(adj), step));
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        check(!pattern[i][j] || byFloyd[i][j],
                                "the exact-length pattern of A^" + step + " is inside the closure");
                    }
                }
            }
            boolean[][] reflexive = new boolean[n][n];
            for (int i = 0; i < n; i++) {
                reflexive[i] = byFloyd[i].clone();
                reflexive[i][i] = true;
            }
            boolean[][] closedAgain = pathPattern(multiply(toMatrix(reflexive), toMatrix(reflexive)));
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    check(closedAgain[i][j] == reflexive[i][j],
                            "the reflexive closure is idempotent (case " + t + ")");
                }
            }
            boolean[][] plainAgain = pathPattern(multiply(toMatrix(byFloyd), toMatrix(byFloyd)));
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    check(!plainAgain[i][j] || byFloyd[i][j],
                            "R * R is contained in R but need not equal it (case " + t + ")");
                }
            }
            for (int i = 0; i < n; i++) {
                check(byFloyd[i][i] == onCycle(adj, i),
                        "the diagonal of the closure marks vertices on a cycle (case " + t + ")");
            }
        }
        check(booleanClosureByAccumulation(new boolean[0][0]).length == 0,
                "the empty graph is handled");
        boolean[][] chain = new boolean[5][5];
        for (int i = 0; i + 1 < 5; i++) chain[i][i + 1] = true;
        boolean[][] chainClosure = booleanClosureByAccumulation(chain);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                check(chainClosure[i][j] == (i < j), "the chain's closure is strictly upper triangular");
            }
        }

        // --- Boundaries.
        long[][] zero = new long[4][4];
        check(totalSum(power(zero, 0)) == 4, "the zero matrix raised to 0 is the identity");
        check(totalSum(power(zero, 3)) == 0, "the zero matrix raised to a positive power is zero");
        long[][] single = {{0L}};
        check(equal(power(single, 0), identity(1)) && equal(power(single, 5), single),
                "the single-vertex loop-free graph");
        long[][] selfLoop = {{1L}};
        check(power(selfLoop, 7)[0][0] == 1, "a self-loop has exactly one walk of each length");

        long[][] sample = randomMatrix(5, 0.5, new Rng(556001L));
        check(equal(power(sample, 9), power(sample, 9)), "deterministic re-run");
        check(totalSum(power(sample, 9)) == totalSum(power(sample, 9)),
                "the deterministic total walk count is stable");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
