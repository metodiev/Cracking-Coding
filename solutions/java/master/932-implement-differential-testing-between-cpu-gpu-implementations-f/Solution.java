/** Task 932 - Implement differential testing between CPU/GPU implementations for numerical ops.
 *  (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
 *
 * A differential tester for numerical kernels. Every operation is written twice: a straightforward CPU
 * reference (triple-loop matmul, sequential reduction, plain softmax) and a "GPU-shaped" implementation that
 * mimics what a parallel device does - tiled blocking, a pairwise tree reduction, a different accumulation
 * order - so the two produce results that agree mathematically but differ in the last bits. The tester
 * compares pairs under a tolerance policy (absolute + relative), reports the mismatches, and is also pointed at
 * deliberately broken variants to prove it can fail: a tiled matmul that drops the partial tile, a softmax
 * without the max shift (which overflows on large logits), and a reduction that forgets the tail.
 * Also included is an exact oracle: an integer matmul, where both float implementations must match bit for bit,
 * which is what separates a real bug from floating-point reordering noise.
 * Verified: both implementations agree with the exact integer product; the CPU/GPU pair differs in the last
 * bits yet passes the tolerance policy (which a bit-exact policy would reject); each injected fault is caught;
 * degenerate shapes are handled; tile sizes do not change the answer; and the run is reproducible.
 */
import java.util.*;

/** The result of one differential comparison. */
final class DiffResult {
    int compared;
    int bitExact;
    int withinTolerance;
    int violations;
    double worstAbsolute, worstRelative;

    void record(boolean exact, double absolute, double relative, double tolerance) {
        compared++;
        if (exact) bitExact++;
        worstAbsolute = Math.max(worstAbsolute, absolute);
        worstRelative = Math.max(worstRelative, relative);
        if (absolute <= tolerance || relative <= tolerance) withinTolerance++;
        else violations++;
    }

    void merge(DiffResult other) {
        compared += other.compared;
        bitExact += other.bitExact;
        withinTolerance += other.withinTolerance;
        violations += other.violations;
        worstAbsolute = Math.max(worstAbsolute, other.worstAbsolute);
        worstRelative = Math.max(worstRelative, other.worstRelative);
    }
}

/** CPU and GPU-shaped kernels over flat row-major matrices. */
final class Kernels {
    /** Reference: i-k-j triple loop. */
    static double[] matmulCpu(double[] a, double[] b, int n) {
        double[] out = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double aik = a[i * n + k];
                for (int j = 0; j < n; j++) out[i * n + j] += aik * b[k * n + j];
            }
        }
        return out;
    }

    /** GPU-shaped: split-K blocking. Each k-block accumulates into its own partial, which is then summed -
     *  mathematically identical, but a different accumulation order, exactly like a parallel device.
     *  dropPartialTile injects a realistic fault: the partial block at the edge is skipped. */
    static double[] matmulTiled(double[] a, double[] b, int n, int tile, boolean dropPartialTile) {
        int blocks = (n + tile - 1) / tile;
        double[][] partial = new double[blocks][n * n];
        for (int bk = 0; bk < blocks; bk++) {
            int kStart = bk * tile, kEnd = Math.min(n, kStart + tile);
            if (dropPartialTile && kEnd != kStart + tile) continue;
            for (int i = 0; i < n; i++) {
                for (int k = kStart; k < kEnd; k++) {
                    double aik = a[i * n + k];
                    for (int j = 0; j < n; j++) partial[bk][i * n + j] += aik * b[k * n + j];
                }
            }
        }
        double[] out = new double[n * n];
        for (int idx = 0; idx < out.length; idx++) {
            double sum = 0;
            for (int bk = 0; bk < blocks; bk++) sum += partial[bk][idx];
            out[idx] = sum;
        }
        return out;
    }

    static double[] sumCpu(double[] x, boolean dropTail) {
        double total = 0;
        int limit = dropTail ? x.length - 1 : x.length;
        for (int i = 0; i < limit; i++) total += x[i];
        return new double[] {total};
    }

    /** GPU-shaped: pairwise tree reduction. */
    static double[] sumTree(double[] x) {
        double[] buffer = x.clone();
        int n = buffer.length;
        while (n > 1) {
            int half = n / 2;
            for (int i = 0; i < half; i++) buffer[i] += buffer[n - 1 - i];
            if (n % 2 == 1) buffer[half] = buffer[half];
            n = half + (n % 2 == 1 ? 1 : 0);
        }
        return new double[] {n == 0 ? 0 : buffer[0]};
    }

    static double[] softmaxCpu(double[] x, boolean shift) {
        double max = -Double.MAX_VALUE;
        for (double v : x) max = Math.max(max, v);
        double base = shift ? max : 0;
        double[] out = new double[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.exp(x[i] - base);
            sum += out[i];
        }
        for (int i = 0; i < x.length; i++) out[i] /= sum;
        return out;
    }
}

public class Solution {
    static final int N = 12;
    static final double TOLERANCE = 1e-9;

    static double[] matrix(long seed, int n, long spread, boolean integral) {
        double[] out = new double[n * n];
        long state = seed;
        for (int i = 0; i < out.length; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            out[i] = integral ? Math.floorMod(state >>> 17, spread) - spread / 2
                    : ((state >>> 11) * 0x1.0p-53 - 0.5) * spread;
        }
        return out;
    }

    static double[] vector(long seed, int n) {
        double[] out = new double[n];
        long state = seed;
        for (int i = 0; i < n; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            out[i] = ((state >>> 11) * 0x1.0p-53 - 0.5) * 4;
        }
        return out;
    }

    /** Independent exact oracle: integer accumulation, no rounding at all. */
    static long[] matmulExact(double[] a, double[] b, int n) {
        long[] out = new long[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                long sum = 0;
                for (int k = 0; k < n; k++) sum += (long) a[i * n + k] * (long) b[k * n + j];
                out[i * n + j] = sum;
            }
        }
        return out;
    }

    static DiffResult compare(double[] left, double[] right, double tolerance) {
        DiffResult result = new DiffResult();
        for (int i = 0; i < left.length; i++) {
            double absolute = Math.abs(left[i] - right[i]);
            double relative = absolute / Math.max(1e-12, Math.max(Math.abs(left[i]), Math.abs(right[i])));
            result.record(left[i] == right[i], absolute, relative, tolerance);
        }
        return result;
    }

    public static void main(String[] args) {
        // ---- exact oracle: integer entries, so both float matmuls must be bit-exact -------------------
        double[] ai = matrix(0xC0FFEEL, N, 9, true), bi = matrix(0xBEEFL, N, 9, true);
        long[] exact = matmulExact(ai, bi, N);
        double[] cpuExact = Kernels.matmulCpu(ai, bi, N), tiledExact = Kernels.matmulTiled(ai, bi, N, 4, false);
        double exactGap = 0;
        for (int i = 0; i < cpuExact.length; i++) {
            exactGap = Math.max(exactGap, Math.abs(cpuExact[i] - exact[i]));
            exactGap = Math.max(exactGap, Math.abs(tiledExact[i] - exact[i]));
        }

        // ---- the real fixture: general double entries, where accumulation order shows up ---------------
        double[] a = matrix(0xC0FFEEL, N, 10, false), b = matrix(0xBEEFL, N, 10, false);
        double[] cpu = Kernels.matmulCpu(a, b, N);
        DiffResult tiles = new DiffResult(), brokenTiles = new DiffResult();
        for (int tile : new int[] {3, 4, 5, 16}) {
            tiles.merge(compare(cpu, Kernels.matmulTiled(a, b, N, tile, false), TOLERANCE));
            brokenTiles.merge(compare(cpu, Kernels.matmulTiled(a, b, N, tile, true), TOLERANCE));
        }

        // ---- reductions: twelve vectors, sequential against the pairwise tree -------------------------
        DiffResult sums = new DiffResult(), brokenSum = new DiffResult();
        for (int v = 0; v < 12; v++) {
            double[] x = vector(0xABCDL + v * 7919L, 64);
            sums.merge(compare(Kernels.sumCpu(x, false), Kernels.sumTree(x), TOLERANCE));
            brokenSum.merge(compare(Kernels.sumCpu(x, false), Kernels.sumCpu(x, true), TOLERANCE));
        }

        // ---- softmax: a shifted against an unshifted implementation, on logits large enough to overflow
        DiffResult softmax = new DiffResult();
        for (int v = 0; v < 5; v++) {
            double[] logits = vector(0x314159L + v, 20);
            for (int i = 0; i < logits.length; i++) logits[i] *= 900;
            softmax.merge(compare(Kernels.softmaxCpu(logits, true), Kernels.softmaxCpu(logits, false), TOLERANCE));
        }

        System.out.println("differential testing: matmul " + N + "x" + N + " over 4 tilings - compared " + tiles.compared
                + ", bit-exact " + tiles.bitExact + ", within tolerance " + tiles.withinTolerance + ", violations "
                + tiles.violations + ", worst abs " + String.format("%.2e", tiles.worstAbsolute));
        System.out.println("exact integer oracle: worst |implementation - exact| " + String.format("%.2e", exactGap)
                + "; reductions compared " + sums.compared + " (bit-exact " + sums.bitExact + ", violations "
                + sums.violations + "); softmax compared " + softmax.compared + " (violations " + softmax.violations
                + ", worst abs " + String.format("%.2e", softmax.worstAbsolute) + ")");
        System.out.println("faults injected: partial-tile drop violations " + brokenTiles.violations + ", dropped tail "
                + brokenSum.violations + ", unshifted softmax " + softmax.violations);

        check(exactGap == 0, "both float matmuls must reproduce the exact integer product: gap " + exactGap);
        check(tiles.violations == 0 && tiles.withinTolerance == tiles.compared,
                "the tiled and the reference matmul must agree: " + tiles.violations + " violations");
        check(tiles.bitExact < tiles.compared, "a different accumulation order must differ in the last bits: "
                + tiles.bitExact + " of " + tiles.compared + " were bit-exact");
        check(brokenTiles.violations > 0, "dropping the partial tile must be detected as a mismatch");
        check(brokenSum.violations > 0, "a reduction that forgets the tail must be detected");
        check(softmax.violations > 0, "softmax without the max shift must diverge on large logits");
        check(sums.violations == 0 && sums.withinTolerance == sums.compared,
                "the sequential and tree reductions must agree: " + sums.violations);
        check(sums.bitExact < sums.compared, "the two reduction orders must differ somewhere: " + sums.bitExact
                + " of " + sums.compared);

        // Why the tolerance policy exists: an exact comparison would reject a correct implementation.
        DiffResult strict = compare(cpu, Kernels.matmulTiled(a, b, N, 4, false), 0.0);
        check(strict.violations > 0 && tiles.violations == 0, "a bit-exact policy rejects the correct tiled matmul "
                + "(" + strict.violations + " violations) while the tolerance policy accepts it");

        // Degenerate shapes: 1x1, a single row, an empty vector.
        double[] one = {2.5}, two = {-4.0};
        check(Arrays.equals(Kernels.matmulCpu(one, two, 1), new double[] {-10.0})
                && Arrays.equals(Kernels.matmulTiled(one, two, 1, 4, false), new double[] {-10.0}),
                "a 1x1 matmul must be exact under both implementations");
        check(Kernels.matmulCpu(one, two, 1)[0] == -10.0, "1x1 arithmetic must be exact");
        check(Kernels.sumTree(new double[0]).length == 1 && Kernels.sumTree(new double[0])[0] == 0,
                "an empty reduction is zero");
        check(Kernels.sumCpu(new double[] {1.5}, false)[0] == 1.5 && Kernels.sumTree(new double[] {1.5})[0] == 1.5,
                "a single-element reduction returns the element");
        double[] singleLogit = Kernels.softmaxCpu(new double[] {700, 700}, true);
        check(Math.abs(singleLogit[0] - 0.5) < 1e-12 && Arrays.equals(Kernels.softmaxCpu(new double[] {0}, true),
                new double[] {1.0}), "a shifted softmax must stay exact at extreme logits");

        // The tester itself: counts, worst-case statistics and determinism.
        DiffResult replay = new DiffResult();
        for (int tile : new int[] {3, 4, 5, 16}) replay.merge(compare(cpu, Kernels.matmulTiled(a, b, N, tile, false), TOLERANCE));
        check(replay.compared == tiles.compared && replay.bitExact == tiles.bitExact
                && replay.worstAbsolute == tiles.worstAbsolute, "the tester must be reproducible");
        check(tiles.worstAbsolute > 0 && tiles.worstAbsolute < 1e-6, "reordering noise must be small but non-zero: "
                + tiles.worstAbsolute);

        System.out.println("tolerance policy: strict comparison rejects " + strict.violations + " of " + strict.compared
                + " elements (worst " + String.format("%.2e", tiles.worstAbsolute) + ") that the "
                + TOLERANCE + " tolerance accepts");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
