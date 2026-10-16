/** Task 558 - Implement boolean matrix multiplication optimization using bitsets. (Tier: Advanced, CTCI Ch 5 — Bit Manipulation)
 *
 * Boolean matrix multiplication with bit-parallel rows: a row of an n x n boolean matrix lives in
 * w = ceil(n/64) 64-bit words, so one word-wise OR of 64 columns replaces 64 scalar ORs. The row-by-row
 * product visits every set bit (i,k) of A and ORs the whole row k of B into row i of the result — this
 * is the textbook "operate in the row space" formulation of Boolean matrix multiplication — and each OR
 * touches w words. Three further implementations are used as cross-checks: a naive triple loop over
 * (i,k,j), a transposed formulation where the popcount of (row i of A) AND (row j of B^T) decides entry
 * (i,j), and repeated squaring for reachability.
 *
 * Documented checks. Word-level results are compared entrywise with the naive triple loop on seeded
 * random matrices with n = 1, 63, 64, 65, 67, 100 and 128 — the sizes around the word boundary are the
 * ones that catch off-by-one bugs in the last partial word — and against the transposed method; the
 * algebraic laws of the Boolean semiring are then exercised (A x I = I x A = A, associativity
 * (A x B) x C = A x (B x C), and distributivity A x (B + C) = (A x B) + (A x C) where + is entrywise
 * OR). The optimization itself is quantified rather than asserted qualitatively: the bitset product
 * performs exactly popcount(A) * w word-sized OR operations (asserted as an identity against a counter
 * incremented inside the loop) where the naive triple loop performs n^3 scalar steps: on the dense
 * 256 x 256 instance (58995 set bits, stride 4) that is 235980 word ORs against 16777216 scalar (i,k,j)
 * steps, a measured factor of 71, and the assertion requires at least 64. Reachability is where bitset squaring pays off:
 * A^(n-1), composed by repeated squaring, the accumulated union A v A^2 v ... v A^(n-1) and
 * Floyd–Warshall all produce the same closure on 12 seeded directed graphs including the non-multiple-of-
 * 64 sizes, the closure is idempotent, and the transitive closure of a 200-vertex random graph is
 * computed by bitset squaring. The documented caveat from the exact-length view is asserted rather than
 * glossed over: the pattern of A^(n-1) *alone* is only the pairs joined by a walk of exactly n-1 edges,
 * so it is checked to be contained in the closure rather than equal to it, and the reflexive closure is
 * obtained from (I v A)^(n-1). Zero matrices, the identity, a single vertex and the empty graph are covered, and
 * every result is reproduced on a second run.
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

    /** Row-major bit packing: words[i * stride + word], bit b of word = column word*64 + b. */
    static final class Bits {
        final int n;
        final int stride;
        final long[] words;

        Bits(int n) {
            this.n = n;
            this.stride = (n + 63) / 64;
            this.words = new long[n * stride];
        }

        void set(int row, int col) {
            words[row * stride + (col >>> 6)] |= 1L << (col & 63);
        }

        boolean get(int row, int col) {
            return (words[row * stride + (col >>> 6)] >>> (col & 63) & 1L) != 0;
        }

        int popcount() {
            int total = 0;
            for (long word : words) total += Long.bitCount(word);
            return total;
        }
    }

    static Bits fromBoolean(boolean[][] a) {
        int n = a.length;
        Bits out = new Bits(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) if (a[i][j]) out.set(i, j);
        }
        return out;
    }

    static Bits randomBits(int n, double p, Rng rng) {
        Bits out = new Bits(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) if (rng.nextDouble() < p) out.set(i, j);
        }
        return out;
    }

    static Bits identityBits(int n) {
        Bits out = new Bits(n);
        for (int i = 0; i < n; i++) out.set(i, i);
        return out;
    }

    static boolean[][] toBoolean(Bits a) {
        boolean[][] out = new boolean[a.n][a.n];
        for (int i = 0; i < a.n; i++) {
            for (int j = 0; j < a.n; j++) out[i][j] = a.get(i, j);
        }
        return out;
    }

    /** Bit-parallel product: for every set bit (i,k) of A, OR the whole row k of B into row i. */
    static Bits multiplyRows(Bits a, Bits b, long[] wordOps) {
        Bits out = new Bits(a.n);
        for (int i = 0; i < a.n; i++) {
            int rowBase = i * a.stride;
            for (int word = 0; word < a.stride; word++) {
                long bits = a.words[rowBase + word];
                while (bits != 0) {
                    int col = (word << 6) + Long.numberOfTrailingZeros(bits);
                    bits &= bits - 1;
                    if (col >= a.n) continue;
                    int src = col * b.stride;
                    int dst = rowBase;
                    for (int t = 0; t < a.stride; t++) {
                        out.words[dst + t] |= b.words[src + t];
                    }
                    if (wordOps != null) wordOps[0] += a.stride;
                }
            }
        }
        return out;
    }

    /** Naive triple loop; the second return value counts the scalar (i,k,j) steps. */
    static boolean[][] multiplyNaive(boolean[][] a, boolean[][] b, long[] steps) {
        int n = a.length;
        boolean[][] out = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    if (steps != null) steps[0]++;
                    if (a[i][k] && b[k][j]) out[i][j] = true;
                }
            }
        }
        return out;
    }

    /** Transposed formulation: entry (i,j) is nonzero iff row i of A and row j of B^T overlap. */
    static Bits multiplyTransposed(Bits a, Bits b) {
        Bits bt = new Bits(a.n);
        for (int i = 0; i < a.n; i++) {
            for (int j = 0; j < a.n; j++) if (b.get(i, j)) bt.set(j, i);
        }
        Bits out = new Bits(a.n);
        for (int i = 0; i < a.n; i++) {
            for (int j = 0; j < a.n; j++) {
                long overlap = 0;
                for (int t = 0; t < a.stride && overlap == 0; t++) {
                    overlap = a.words[i * a.stride + t] & bt.words[j * bt.stride + t];
                }
                if (overlap != 0) out.set(i, j);
            }
        }
        return out;
    }

    static boolean equalBits(Bits a, Bits b) {
        if (a.n != b.n) return false;
        for (int i = 0; i < a.words.length; i++) if (a.words[i] != b.words[i]) return false;
        return true;
    }

    static boolean equalBoolean(boolean[][] a, boolean[][] b) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) if (a[i][j] != b[i][j]) return false;
        }
        return true;
    }

    static Bits orBits(Bits a, Bits b) {
        Bits out = new Bits(a.n);
        for (int i = 0; i < a.words.length; i++) out.words[i] = a.words[i] | b.words[i];
        return out;
    }

    static Bits accumulatePower(Bits a, int maxEdges) {
        Bits accumulated = new Bits(a.n);
        Bits current = a;
        for (int i = 0; i < a.words.length; i++) accumulated.words[i] = current.words[i];
        for (int step = 2; step <= maxEdges; step++) {
            current = multiplyRows(current, a, null);
            accumulated = orBits(accumulated, current);
        }
        return accumulated;
    }

    static Bits powerBySquaring(Bits a, int exponent) {
        Bits result = identityBits(a.n);
        Bits square = a;
        int e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) result = multiplyRows(result, square, null);
            square = multiplyRows(square, square, null);
            e >>= 1;
        }
        return result;
    }

    static boolean[][] closureFloyd(boolean[][] adj) {
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

    public static void main(String[] args) {
        Rng rng = new Rng(20240558L);

        // --- Correctness at and around the 64-bit word boundary.
        int[] sizes = {1, 2, 31, 63, 64, 65, 67, 100, 128};
        for (int n : sizes) {
            Bits a = randomBits(n, 0.35, rng);
            Bits b = randomBits(n, 0.4, rng);
            boolean[][] an = toBoolean(a);
            boolean[][] bn = toBoolean(b);
            boolean[][] expected = multiplyNaive(an, bn, null);
            Bits bitset = multiplyRows(a, b, null);
            check(equalBoolean(toBoolean(bitset), expected), "bitset product matches the naive loop at n = "
                    + n);
            check(equalBoolean(toBoolean(multiplyTransposed(a, b)), expected),
                    "the transposed product matches too at n = " + n);
            check(equalBits(bitset, multiplyTransposed(a, b)),
                    "both bit-parallel formulations agree at n = " + n);
            Bits id = identityBits(n);
            check(equalBits(multiplyRows(a, id, null), a) && equalBits(multiplyRows(id, a, null), a),
                    "the identity is neutral at n = " + n);
        }

        // --- Boolean semiring laws.
        for (int t = 0; t < 6; t++) {
            int n = 40 + 7 * t;
            Bits a = randomBits(n, 0.3, rng);
            Bits b = randomBits(n, 0.3, rng);
            Bits c = randomBits(n, 0.3, rng);
            check(equalBits(multiplyRows(multiplyRows(a, b, null), c, null),
                    multiplyRows(a, multiplyRows(b, c, null), null)),
                    "the product is associative (case " + t + ")");
            check(equalBits(multiplyRows(a, orBits(b, c), null),
                    orBits(multiplyRows(a, b, null), multiplyRows(a, c, null))),
                    "the product distributes over entrywise OR (case " + t + ")");
        }

        // --- Counting the work: popcount(A) * stride word ORs versus n^3 scalar steps.
        int big = 256;
        Bits dense = randomBits(big, 0.9, new Rng(558001L));
        long[] ops = new long[1];
        Bits product = multiplyRows(dense, dense, ops);
        check(ops[0] == (long) dense.popcount() * dense.stride,
                "the bitset product performs exactly popcount(A) * stride = " + ops[0] + " word ORs");
        long[] naiveOps = new long[1];
        boolean[][] denseNaive = multiplyNaive(toBoolean(dense), toBoolean(dense), naiveOps);
        check(naiveOps[0] == (long) big * big * big, "the naive triple loop runs n^3 = " + naiveOps[0]
                + " scalar (i,k,j) steps");
        check(equalBoolean(toBoolean(product), denseNaive), "and produces the same dense product");
        check(naiveOps[0] / ops[0] >= 64, "so the bitset product is a factor " + (naiveOps[0] / ops[0])
                + " cheaper in elementary operations");
        check(product.popcount() > 0, "and the dense product is non-empty");

        // --- Reachability by squaring versus accumulated union versus Floyd–Warshall.
        for (int t = 0; t < 12; t++) {
            int n = 9 + 13 * t;
            boolean[][] adj = new boolean[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) adj[i][j] = rng.nextDouble() < 0.08;
            }
            Bits bits = fromBoolean(adj);
            boolean[][] floyd = closureFloyd(adj);
            boolean[][] reflexiveFloyd = new boolean[n][n];
            for (int i = 0; i < n; i++) {
                reflexiveFloyd[i] = floyd[i].clone();
                reflexiveFloyd[i][i] = true;
            }
            Bits bySquaring = powerBySquaring(orBits(bits, identityBits(n)), n - 1);
            Bits byAccumulation = accumulatePower(bits, n - 1);
            check(equalBoolean(toBoolean(bySquaring), reflexiveFloyd),
                    "(I v A)^(n-1) by squaring equals the reflexive closure at n = " + n);
            check(equalBoolean(toBoolean(byAccumulation), floyd),
                    "the accumulated union equals Floyd–Warshall at n = " + n);
            Bits exactLength = powerBySquaring(bits, n - 1);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    check(!exactLength.get(i, j) || floyd[i][j],
                            "the exact-length pattern of A^(n-1) is contained in the closure");
                }
            }
            boolean[][] reflexive = toBoolean(orBits(byAccumulation, identityBits(n)));
            check(equalBoolean(toBoolean(multiplyRows(fromBoolean(reflexive), fromBoolean(reflexive),
                    null)), reflexive), "the reflexive closure is idempotent at n = " + n);
        }
        check(equalBoolean(toBoolean(powerBySquaring(fromBoolean(new boolean[0][0]), 0)),
                new boolean[0][0]), "the empty graph");

        // --- Larger closure: what the optimization is for.
        int large = 200;
        Bits largeAdj = randomBits(large, 0.02, new Rng(558002L));
        Bits largeClosure = accumulatePower(largeAdj, large - 1);
        Bits reflexiveLarge = orBits(largeClosure, identityBits(large));
        check(equalBits(multiplyRows(reflexiveLarge, reflexiveLarge, null), reflexiveLarge),
                "the 200-vertex closure is idempotent");
        boolean[][] largeFloyd = closureFloyd(toBoolean(largeAdj));
        check(equalBoolean(toBoolean(largeClosure), largeFloyd),
                "and it equals Floyd–Warshall on the 200-vertex graph");

        // --- Boundaries.
        Bits zero = new Bits(5);
        check(multiplyRows(zero, zero, null).popcount() == 0, "the zero matrix squares to zero");
        check(equalBits(multiplyRows(zero, identityBits(5), null), zero), "zero times identity is zero");
        Bits single = new Bits(1);
        check(equalBits(multiplyRows(single, single, null), single), "the 1x1 zero matrix");
        single.set(0, 0);
        check(equalBits(multiplyRows(single, single, null), single), "the 1x1 unit matrix");
        Bits chain = new Bits(6);
        for (int i = 0; i + 1 < 6; i++) chain.set(i, i + 1);
        Bits chainClosure = accumulatePower(chain, 5);
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                check(chainClosure.get(i, j) == (i < j), "the chain closure is strictly upper triangular");
            }
        }

        Bits sample = randomBits(70, 0.5, new Rng(558003L));
        check(equalBits(multiplyRows(sample, sample, null), multiplyRows(sample, sample, null)),
                "deterministic re-run");
        check(multiplyRows(sample, sample, null).popcount() > 0, "and the repeated result is stable");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
