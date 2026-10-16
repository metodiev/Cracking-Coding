/** Task 696 - Implement compensated arithmetic for numerical stability (Kahan). (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Four summation strategies are implemented side by side:
 *
 *   naive       s += x, the rounding error of every step is simply lost;
 *   Kahan       a running compensation c recovers the low-order part of each step, subtracts it
 *               from the next addend and re-extracts what is left: -(c = (s - t) - y) style;
 *   Neumaier    same idea but the compensation is chosen from whichever of s and x dominates,
 *               which fixes the case Kahan cannot handle - a final huge addend that cancels
 *               everything accumulated so far ([1e100, 1, -1e100] is the textbook example);
 *   pairwise    recursive halving, with an error growing like log n instead of n.
 *
 * The exact answers used by the checks are not "the same algorithm run in double": the integer
 * cases are summed exactly in long arithmetic (all inputs and the exact sum are representable),
 * the offset-variance case uses the closed form (n^2-1)/12, and the ill-conditioned examples are
 * chosen so the exact result is a small integer that is analytically known.
 *
 * Also verified: Welford's online variance stays exact where the one-pass E[X^2]-E[X]^2 formula
 * collapses under cancellation, compensated dot products behave like compensated sums, and the
 * compensated results are never worse than the naive ones on the constructed cases.
 */
import java.util.*;

public class Solution {
    private static long prngState = 0xDEADBEEFL;

    static long nextLong() {
        prngState += 0x9E3779B97F4A7C15L;
        long z = prngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), (long) bound);
    }

    static double naiveSum(double[] values) {
        double sum = 0;
        for (double value : values) sum += value;
        return sum;
    }

    static double kahanSum(double[] values) {
        double sum = 0;
        double compensation = 0;
        for (double value : values) {
            double y = value - compensation;
            double t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        return sum;
    }

    static double neumaierSum(double[] values) {
        double sum = 0;
        double compensation = 0;
        for (double value : values) {
            double t = sum + value;
            if (Math.abs(sum) >= Math.abs(value)) {
                compensation += (sum - t) + value;
            } else {
                compensation += (value - t) + sum;
            }
            sum = t;
        }
        return sum + compensation;
    }

    static double pairwiseSum(double[] values) {
        return pairwiseSum(values, 0, values.length);
    }

    private static double pairwiseSum(double[] values, int from, int to) {
        int length = to - from;
        if (length <= 8) {
            double sum = 0;
            for (int i = from; i < to; i++) sum += values[i];
            return sum;
        }
        int middle = from + length / 2;
        return pairwiseSum(values, from, middle) + pairwiseSum(values, middle, to);
    }

    static double compensatedDot(double[] a, double[] b) {
        double sum = 0;
        double compensation = 0;
        for (int i = 0; i < a.length; i++) {
            double product = a[i] * b[i];
            double t = sum + product;
            if (Math.abs(sum) >= Math.abs(product)) {
                compensation += (sum - t) + product;
            } else {
                compensation += (product - t) + sum;
            }
            sum = t;
        }
        return sum + compensation;
    }

    static double naiveDot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    /** Welford's online mean and variance. */
    static double[] welfordVariance(double[] values) {
        long count = 0;
        double mean = 0;
        double m2 = 0;
        for (double value : values) {
            count++;
            double delta = value - mean;
            mean += delta / count;
            m2 += delta * (value - mean);
        }
        return new double[]{mean, m2 / count};
    }

    /** The one-pass formula E[X^2] - E[X]^2, which cancels catastrophically for offset data. */
    static double onePassVariance(double[] values) {
        double sum = 0;
        double squares = 0;
        for (double value : values) {
            sum += value;
            squares += value * value;
        }
        double mean = sum / values.length;
        return squares / values.length - mean * mean;
    }

    public static void main(String[] args) {
        // 1. The canonical cancellation example: 1e16 + 999 ones - 1e16 = 999 exactly.
        double[] ones = new double[1001];
        ones[0] = 1e16;
        for (int i = 1; i <= 999; i++) ones[i] = 1.0;
        ones[1000] = -1e16;
        check(naiveSum(ones) == 0.0, "the naive sum loses every one: " + naiveSum(ones));
        check(neumaierSum(ones) == 999.0, "Neumaier recovers the exact answer: " + neumaierSum(ones));
        check(Math.abs(kahanSum(ones) - 999.0) <= 1.0,
                "Kahan is within one unit of the exact answer: " + kahanSum(ones));
        check(Math.abs(pairwiseSum(ones) - 999.0) > Math.abs(neumaierSum(ones) - 999.0),
                "pairwise summation still loses a few ones: " + pairwiseSum(ones));
        check(Math.abs(kahanSum(ones) - 999.0) < Math.abs(naiveSum(ones) - 999.0),
                "compensation beats naive summation");

        // 2. The documented Kahan failure: a final addend bigger than everything before it.
        double[] lateCancel = {1e100, 1.0, -1e100};
        check(kahanSum(lateCancel) == 0.0, "Kahan genuinely fails on this input: " + kahanSum(lateCancel));
        check(neumaierSum(lateCancel) == 1.0, "Neumaier's branching recovers it: " + neumaierSum(lateCancel));
        check(naiveSum(lateCancel) == 0.0 && pairwiseSum(lateCancel) == 0.0,
                "the other two strategies fail as well");

        // 3. A three-addend variant: exact answer 3.
        double[] mixed = {1e16, 1.0, 1.0, 1.0, -1e16};
        check(naiveSum(mixed) == 0.0 && neumaierSum(mixed) == 3.0,
                "exact 3 recovered: " + neumaierSum(mixed));
        check(Math.abs(kahanSum(mixed) - 3.0) <= 1.0, "Kahan is close: " + kahanSum(mixed));

        // 4. Random integer data with two huge cancelling terms: the exact long sum is the oracle.
        double[] exactValues = new double[1002];
        long exactTotal = 0;
        int compensatedWins = 0;
        for (int trial = 0; trial < 20; trial++) {
            exactValues[0] = 1e16;
            exactValues[1001] = -1e16;
            exactTotal = 0;
            for (int i = 1; i <= 1000; i++) {
                int value = nextInt(20001) - 10000;
                exactValues[i] = value;
                exactTotal += value;
            }
            double naive = naiveSum(exactValues);
            double neumaier = neumaierSum(exactValues);
            double naiveError = Math.abs(naive - exactTotal);
            double compensatedError = Math.abs(neumaier - exactTotal);
            if (compensatedError <= naiveError) compensatedWins++;
            if (compensatedError > 2.0) {
                check(false, "compensated error too large: " + compensatedError);
            }
        }
        check(compensatedWins == 20, "Neumaier is never worse than naive on 20 random cases");
        check(Math.abs(naiveSum(exactValues) - exactTotal) > 0.5,
                "the naive sum really is off by more than half: " + naiveSum(exactValues));

        // 5. Offset variance: the one-pass formula collapses, Welford and the shifted form do not.
        double[] offset = new double[1000];
        for (int i = 0; i < offset.length; i++) offset[i] = 1.0e9 + i;
        double exactVariance = (999.0 * 1001.0) / 12.0;
        double onePass = onePassVariance(offset);
        double[] welford = welfordVariance(offset);
        check(Math.abs(onePass - exactVariance) > 1.0,
                "the one-pass variance is badly wrong: " + onePass + " vs " + exactVariance);
        check(Math.abs(welford[1] - exactVariance) < 1e-9,
                "Welford's variance is exact: " + welford[1]);
        check(Math.abs(welford[0] - 1000000499.5) < 1e-6, "Welford's mean is right: " + welford[0]);
        double shifted = onePassVariance(shifted(offset, 1.0e9));
        check(Math.abs(shifted - exactVariance) < 1e-6,
                "centring the data first also fixes it: " + shifted);

        // 6. Compensated dot product behaves like the compensated sum.
        double[] left = {1e16, 1.0, 1.0, 1.0, -1e16};
        double[] right = {1.0, 1.0, 1.0, 1.0, 1.0};
        check(naiveDot(left, right) == 0.0, "the naive dot product loses the exact 3");
        check(compensatedDot(left, right) == 3.0,
                "the compensated dot product is exact: " + compensatedDot(left, right));
        double[] bigLeft = new double[202];
        double[] bigRight = new double[202];
        bigLeft[0] = 1e16;
        bigLeft[201] = -1e16;
        bigRight[0] = 1.0;
        bigRight[201] = 1.0;
        long exactDot = 0;
        for (int i = 1; i <= 200; i++) {
            int value = nextInt(101) - 50;
            bigLeft[i] = value;
            bigRight[i] = 3.0;
            exactDot += 3L * value;
        }
        check(compensatedDot(bigLeft, bigRight) == exactDot,
                "compensated dot product matches the long oracle: " + compensatedDot(bigLeft, bigRight));
        check(Math.abs(naiveDot(bigLeft, bigRight) - exactDot) > 2.0,
                "the naive dot product does not: " + naiveDot(bigLeft, bigRight));

        // 7. Ordering: pairwise summation has an error that grows like log n, not n.
        double[] longRun = new double[4096];
        for (int i = 0; i < longRun.length; i++) longRun[i] = 0.1;
        double expected = 4096 * 0.1;
        check(Math.abs(pairwiseSum(longRun) - expected) < Math.abs(naiveSum(longRun) - expected),
                "pairwise is at least as accurate as naive on a long run: "
                        + pairwiseSum(longRun) + " vs " + naiveSum(longRun));
        check(Math.abs(neumaierSum(longRun) - expected) < 1e-9,
                "Neumaier nails the 4096 x 0.1 sum: " + neumaierSum(longRun));

        // 8. Degenerate inputs.
        check(naiveSum(new double[0]) == 0.0 && kahanSum(new double[0]) == 0.0
                && neumaierSum(new double[0]) == 0.0 && pairwiseSum(new double[0]) == 0.0,
                "the empty sum is zero for every strategy");
        check(kahanSum(new double[]{42.5}) == 42.5 && neumaierSum(new double[]{42.5}) == 42.5,
                "a single element is returned unchanged");
        double[] cancel = {1e17, -1e17, 1e17, -1e17};
        check(neumaierSum(cancel) == 0.0 && kahanSum(cancel) == 0.0, "exact cancellation stays zero");
        double[] zeros = new double[64];
        check(naiveSum(zeros) == 0 && pairwiseSum(zeros) == 0 && compensatedDot(zeros, zeros) == 0,
                "all-zero inputs stay zero");

        // 9. Determinism of the whole suite.
        prngState = 0x55AAL;
        double[] firstRun = randomArray(50);
        prngState = 0x55AAL;
        double[] secondRun = randomArray(50);
        check(Arrays.equals(firstRun, secondRun) && kahanSum(firstRun) == kahanSum(secondRun),
                "the generator and the compensated sums are deterministic");

        System.out.println("PASS");
    }

    static double[] randomArray(int length) {
        double[] values = new double[length];
        for (int i = 0; i < length; i++) values[i] = (nextInt(2000001) - 1000000) / 1000.0;
        return values;
    }

    static double[] shifted(double[] values, double offset) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) out[i] = values[i] - offset;
        return out;
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
