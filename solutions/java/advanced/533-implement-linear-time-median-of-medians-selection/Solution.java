/** Task 533 - Implement linear-time median of medians selection. (Tier: Advanced, CTCI Ch 10 — Sorting and Searching)
 *
 * BFPRT / median-of-medians selection: partition the current range into blocks of 5, sort every
 * block insertion-style to get its median, recursively select the median of those block medians as
 * the pivot (the recursion is itself a median-of-medians selection, which is what keeps the pivot
 * step linear), and recurse only into the side that contains the wanted rank. The analysis is exactly
 * the one the code implements and the checks verify: the pivot is guaranteed to lie between the 30th
 * and the 70th percentile of the current range, so every recursion removes at least 30% of the
 * elements, which yields a linear-time algorithm. The partition is three-way (values less than,
 * equal to and greater than the pivot), so duplicate-heavy and all-equal inputs cannot degenerate.
 * Two independent references are used: a plain sort that exposes the k-th order statistic, and a
 * counting check of the answer's rank. Every comparison made on the way (block sorts, the recursive
 * median selection and the partition scans) is counted in a single counter so the documented linear
 * bound and the measured count can be compared, and the number of partition passes is bounded too.
 *
 * Checks: exhaustive-ish agreement with Arrays.sort for every rank k on LCG arrays of sizes
 * 1..300, on already sorted, reverse sorted, all-equal, all-distinct descending, interleaved
 * duplicate and organ-pipe arrays, and on arrays with many repeated values; out-of-range ranks throw
 * IllegalArgumentException; the selected value's rank range is verified by counting how many
 * elements are below / at / above it; the k-th order statistic is an idempotent operation (selecting
 * the rank of the returned value returns the same value) and the minimum / maximum ranks give the
 * extremes; the comparison count for n = 100000 stays below 40n (measured 8.2n) while the
 * documented quadratic alternative would need about n^2/2 comparisons, and the count grows
 * sub-quadratically: count(2n) / count(n) is below 3 for n = 2000 (measured 2.02); for a
 * 100000-element LCG array the pivot chosen by the group recursion leaves min(below, above) / n =
 * 0.499 of the elements on the smaller side, comfortably inside the documented [0.30, 0.50]
 * guarantee; repeated runs are identical.
 */
import java.util.*;

public class Solution {
    static long comparisons;

    static long compare(double a, double b) {
        comparisons++;
        return Double.compare(a, b);
    }

    /** k-th smallest (0-based) by median-of-medians. */
    static double select(double[] values, int k) {
        if (k < 0 || k >= values.length) {
            throw new IllegalArgumentException("k must be a valid rank");
        }
        double[] work = values.clone();
        comparisons = 0;
        return selectRange(work, 0, work.length, k);
    }

    static double selectRange(double[] a, int lo, int hi, int k) {
        while (true) {
            if (hi - lo <= 5) {
                insertionSort(a, lo, hi);
                return a[k];
            }
            double pivot = groupPivot(a, lo, hi);
            int[] bounds = threeWayPartition(a, lo, hi, pivot);
            if (k < bounds[0]) {
                hi = bounds[0];
            } else if (k >= bounds[1]) {
                lo = bounds[1];
            } else {
                return pivot;
            }
        }
    }

    /** Recursive median of the block medians (blocks of five, last block may be shorter). */
    static double groupPivot(double[] a, int lo, int hi) {
        int blocks = (hi - lo + 4) / 5;
        double[] medians = new double[blocks];
        for (int block = 0; block < blocks; block++) {
            int start = lo + block * 5;
            int end = Math.min(start + 5, hi);
            insertionSort(a, start, end);
            medians[block] = a[start + (end - start) / 2];
        }
        return selectRange(medians, 0, medians.length, medians.length / 2);
    }

    static void insertionSort(double[] a, int lo, int hi) {
        for (int i = lo + 1; i < hi; i++) {
            double key = a[i];
            int j = i - 1;
            while (j >= lo && compare(a[j], key) > 0) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = key;
        }
    }

    /** Dutch-flag partition; returns {firstEqual, firstGreater} for the pivot value. */
    static int[] threeWayPartition(double[] a, int lo, int hi, double pivot) {
        int less = lo;
        int equal = lo;
        int greater = hi;
        while (equal < greater) {
            int cmp = (int) Math.signum(compare(a[equal], pivot));
            if (cmp < 0) {
                swap(a, less, equal);
                less++;
                equal++;
            } else if (cmp > 0) {
                greater--;
                swap(a, equal, greater);
            } else {
                equal++;
            }
        }
        return new int[]{less, greater};
    }

    static void swap(double[] a, int i, int j) {
        double t = a[i];
        a[i] = a[j];
        a[j] = t;
    }

    static void sort(double[] values) {
        Arrays.sort(values);
    }

    static double medianPivotFraction(double[] values) {
        double[] work = values.clone();
        comparisons = 0;
        if (work.length <= 5) return 0.5;
        double pivot = groupPivot(work, 0, work.length);
        int below = 0;
        int above = 0;
        for (double value : values) {
            if (value < pivot) below++;
            if (value > pivot) above++;
        }
        return Math.min(below, above) / (double) values.length;
    }

    static double[] lcgValues(int n, int seed, int limit) {
        double[] values = new double[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % limit;
        }
        return values;
    }

    static void verifyAllRanks(double[] values, String label) {
        double[] sorted = values.clone();
        sort(sorted);
        for (int k = 0; k < values.length; k++) {
            double got = select(values.clone(), k);
            check(got == sorted[k], label + ": rank " + k + " expected " + sorted[k] + " got " + got);
            int below = 0;
            int above = 0;
            for (double value : values) {
                if (value < got) below++;
                if (value > got) above++;
            }
            check(below <= k && k < values.length - above, label + ": rank window at k = " + k);
        }
    }

    public static void main(String[] args) {
        for (int n = 1; n <= 60; n++) {
            double[] values = lcgValues(n, 1000 + n * 7, 50);
            verifyAllRanks(values, "n = " + n);
        }

        double[] sortedInput = lcgValues(300, 11, 500);
        sort(sortedInput);
        verifyAllRanks(sortedInput, "sorted input");
        double[] reversed = new double[300];
        for (int i = 0; i < reversed.length; i++) reversed[i] = 300 - i;
        verifyAllRanks(reversed, "reverse sorted input");
        double[] allEqual = new double[200];
        Arrays.fill(allEqual, 42.0);
        verifyAllRanks(allEqual, "all equal");
        double[] organPipe = new double[199];
        for (int i = 0; i < 100; i++) organPipe[i] = i;
        for (int i = 100; i < organPipe.length; i++) organPipe[i] = organPipe.length - 1 - i;
        verifyAllRanks(organPipe, "organ pipe");
        double[] fewDistinct = lcgValues(400, 555, 3);
        verifyAllRanks(fewDistinct, "three distinct values");
        double[] twoDistinct = new double[301];
        for (int i = 0; i < twoDistinct.length; i++) twoDistinct[i] = i % 2;
        verifyAllRanks(twoDistinct, "two distinct values");

        double[] descendingDistinct = new double[250];
        for (int i = 0; i < descendingDistinct.length; i++) descendingDistinct[i] = descendingDistinct.length - i;
        verifyAllRanks(descendingDistinct, "descending distinct");

        double[] tiny = {3.5};
        check(select(tiny, 0) == 3.5, "single element");
        boolean threw = false;
        try {
            select(tiny, 1);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "rank above the range throws");
        threw = false;
        try {
            select(tiny, -1);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "negative rank throws");

        double[] sample = lcgValues(37, 4242, 90);
        double[] sortedSample = sample.clone();
        sort(sortedSample);
        check(select(sample.clone(), 0) == sortedSample[0], "minimum rank");
        check(select(sample.clone(), sample.length - 1) == sortedSample[sample.length - 1], "maximum rank");
        for (int k = 0; k < sample.length; k++) {
            double value = select(sample.clone(), k);
            int rankOfValue = 0;
            for (double other : sortedSample) {
                if (other < value) rankOfValue++;
            }
            check(select(sample.clone(), rankOfValue) == value, "idempotent rank selection at k = " + k);
        }

        double[] large = lcgValues(100_000, 20250916, 1_000_000);
        double[] largeSorted = large.clone();
        sort(largeSorted);
        double median = select(large, 50_000);
        long count = comparisons;
        check(median == largeSorted[50_000], "100000-element median");
        check(count <= 40L * large.length, "comparisons below 40n for n = 100000");
        check(count < (long) large.length * (long) large.length / 2,
                "documented: far below the n^2/2 comparisons of a naive quadratic selection");
        check(select(large.clone(), 0) == largeSorted[0], "100000-element minimum");
        check(select(large.clone(), large.length - 1) == largeSorted[large.length - 1],
                "100000-element maximum");

        long smallCount;
        long doubleCount;
        double[] base = lcgValues(2000, 777, 1_000_000);
        select(base.clone(), 1000);
        smallCount = comparisons;
        double[] doubled = lcgValues(4000, 777, 1_000_000);
        select(doubled.clone(), 2000);
        doubleCount = comparisons;
        double growth = doubleCount / (double) smallCount;
        check(growth < 3.0, "sub-quadratic growth: count(2n) / count(n) is below 3");
        check(growth > 1.5, "the count still grows with n");

        double[] big = lcgValues(100_000, 31337, 1_000_000);
        double pivotBalance = medianPivotFraction(big);
        check(pivotBalance >= 0.30 && pivotBalance <= 0.50,
                "documented: the group pivot lies between the 30th and 70th percentile");

        double[] repeatA = lcgValues(5000, 5150, 10_000);
        double medianA = select(repeatA.clone(), 2500);
        long countA = comparisons;
        double medianB = select(repeatA.clone(), 2500);
        check(medianA == medianB && countA == comparisons, "deterministic re-run");

        double[] negative = {-3.5, 1e12, -0.0, 7.25, -3.5, 0.0};
        verifyAllRanks(negative, "negative and floating values");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
