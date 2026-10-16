/** Task 651 - Implement adaptive indexing (database cracking) prototype. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * Database cracking (Kersten/Idreos): no index is built up front; the column is refined *by the
 * queries themselves*. Each probe "cracks" the value region it touches into three ordered pieces
 *
 *     (< pivot)  (== pivot)  (> pivot)
 *
 * so the array physically remembers every pivot it has been cracked at, and a later probe on an
 * already known pivot costs zero comparisons.
 *
 * Invariant maintained after every crack: data[0, crackEnd) is partitioned by the sorted pivot set
 * p0 < p1 < ...; the equality group of p_i occupies [equalStart(i), equalEnd(i)), values below
 * equalStart(0) are < p0, values between equalEnd(i) and equalStart(i+1) lie strictly inside
 * (p_i, p_i+1), and values after equalEnd(k-1) are > p_{k-1}. A range count [lo, hi] is therefore
 * two cracks - crack(lo) then crack(hi) - answered as equalEnd(hi) - equalStart(lo). Data beyond
 * crackEnd is still in insert order and is never assumed to be ordered.
 *
 * Verified: every range count against a brute-force scan, the partition invariant after every
 * crack, crackEnd monotonicity, the cached-pivot property (a repeated probe costs 0 comparisons),
 * the emergent property that cracking at every distinct value fully sorts the column, and that the
 * crack-on-demand workload does strictly less comparison work than "rescan the column per query".
 */
import java.util.*;

class CrackedColumn {
    final int[] data;
    private final int[] pivots = new int[128];
    private final int[] equalStart = new int[128];
    private final int[] equalEnd = new int[128];
    private int pivotCount;
    int crackEnd;          // data[0, crackEnd) is partitioned, data[crackEnd, n) is untouched
    long comparisons;      // element comparisons spent cracking so far
    int cracks;            // number of fresh pivots actually opened

    CrackedColumn(int[] column) {
        data = column.clone();
    }

    /** Binary search for the insertion index of `value` among the known pivots. */
    private int lowerBound(int value) {
        int lo = 0, hi = pivotCount;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (pivots[mid] < value) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    /** Cracks at `pivot` if needed and returns the pivot's index in the (sorted) pivot list. */
    int crack(int pivot) {
        int index = lowerBound(pivot);
        if (index < pivotCount && pivots[index] == pivot) return index;   // cached, 0 comparisons
        // The piece a new pivot belongs to: between its neighbours. The rightmost piece reaches to the
        // end of the column, so the very first crack scans everything once and every later crack only
        // touches the piece it refines.
        int regionLo = index == 0 ? 0 : equalEnd[index - 1];
        int regionHi = index == pivotCount ? data.length : equalStart[index];
        // Dutch-national-flag partition; [regionLo, lt) < pivot, [lt, scan) == pivot, [gt, regionHi) > pivot
        int lt = regionLo, scan = regionLo, gt = regionHi;
        while (scan < gt) {
            int value = data[scan];
            comparisons++;
            if (value < pivot) {
                swap(scan, lt);
                scan++;
                lt++;
            } else if (value > pivot) {
                gt--;
                swap(scan, gt);
            } else {
                scan++;
            }
        }
        for (int j = pivotCount; j > index; j--) {
            pivots[j] = pivots[j - 1];
            equalStart[j] = equalStart[j - 1];
            equalEnd[j] = equalEnd[j - 1];
        }
        pivots[index] = pivot;
        equalStart[index] = lt;
        equalEnd[index] = gt;
        pivotCount++;
        crackEnd = Math.max(crackEnd, regionHi);   // the piece is now partitioned; the cap never shrinks
        cracks++;
        return index;
    }

    private void swap(int a, int b) {
        int t = data[a];
        data[a] = data[b];
        data[b] = t;
    }

    /** Exact count of values in [lo, hi]; requires lo <= hi. */
    int rangeCount(int lo, int hi) {
        if (lo > hi) throw new IllegalArgumentException("lo > hi: " + lo + " > " + hi);
        int lowIndex = crack(lo);
        int highIndex = crack(hi);
        return equalEnd[highIndex] - equalStart[lowIndex];
    }

    /** Returns a human readable violation of the partition invariant, or null when intact. */
    String verifyPartition() {
        if (crackEnd < 0 || crackEnd > data.length) return "crackEnd out of range: " + crackEnd;
        for (int i = 0; i < pivotCount; i++) {
            if (i > 0 && !(pivots[i - 1] < pivots[i])) return "pivots not sorted at " + i;
            if (equalStart[i] > equalEnd[i]) return "empty/negative equality group at " + i;
            if (equalStart[i] < 0 || equalEnd[i] > crackEnd) return "group outside cracked prefix at " + i;
            for (int k = equalStart[i]; k < equalEnd[i]; k++) {
                if (data[k] != pivots[i]) return "group " + i + " holds " + data[k];
            }
        }
        for (int k = 0; k < (pivotCount == 0 ? crackEnd : equalStart[0]); k++) {
            if (!(data[k] < pivots[0])) return "value " + data[k] + " not below first pivot";
        }
        for (int i = 0; i + 1 < pivotCount; i++) {
            for (int k = equalEnd[i]; k < equalStart[i + 1]; k++) {
                if (!(data[k] > pivots[i] && data[k] < pivots[i + 1])) {
                    return "gap " + i + " holds " + data[k];
                }
            }
        }
        if (pivotCount > 0) {
            for (int k = equalEnd[pivotCount - 1]; k < crackEnd; k++) {
                if (!(data[k] > pivots[pivotCount - 1])) return "value " + data[k] + " not above last pivot";
            }
        }
        int covered = 0;
        for (int i = 0; i < pivotCount; i++) covered += equalEnd[i] - equalStart[i];
        for (int i = 0; i + 1 < pivotCount; i++) covered += equalStart[i + 1] - equalEnd[i];
        if (pivotCount > 0) covered += equalStart[0] + (crackEnd - equalEnd[pivotCount - 1]);
        if (covered != crackEnd) return "covered " + covered + " != crackEnd " + crackEnd;
        return null;
    }
}

public class Solution {
    private static long state = 0x9E3779B97F4A7C15L;

    /** splitmix64 - deterministic, no Math.random. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    static int[] randomColumn(int size, int maxValue) {
        int[] column = new int[size];
        for (int i = 0; i < size; i++) column[i] = nextInt(maxValue);
        return column;
    }

    static int bruteRangeCount(int[] column, int lo, int hi) {
        int count = 0;
        for (int value : column) if (value >= lo && value <= hi) count++;
        return count;
    }

    public static void main(String[] args) {
        // --- degenerate inputs -------------------------------------------------
        CrackedColumn empty = new CrackedColumn(new int[0]);
        check(empty.rangeCount(3, 9) == 0, "empty column, range probe");
        check(empty.verifyPartition() == null, "empty column invariant");
        check(empty.crackEnd == 0, "empty column crackEnd");

        CrackedColumn single = new CrackedColumn(new int[]{7});
        check(single.rangeCount(7, 7) == 1, "single: exact hit");
        check(single.rangeCount(0, 6) == 0, "single: below");
        check(single.rangeCount(8, 100) == 0, "single: above");
        check(single.verifyPartition() == null, "single invariant");

        CrackedColumn dup = new CrackedColumn(new int[]{2, 2, 2, 2});
        check(dup.rangeCount(2, 2) == 4, "all-equal column multiplicity");
        check(dup.rangeCount(1, 1) == 0 && dup.rangeCount(3, 3) == 0, "all-equal column neighbours");
        check(dup.verifyPartition() == null, "all-equal invariant");

        // --- rejected input (negative check) ----------------------------------
        boolean rejected = false;
        try {
            dup.rangeCount(5, 4);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "lo > hi must be rejected");

        // --- probe sequence on a small column, cross-checked against brute force
        int[] base = {9, 3, 7, 1, 8, 3, 5, 5, 0, 4, 7, 2, 6, 9, 3};
        CrackedColumn column = new CrackedColumn(base);
        int[] lows = {3, 5, 0, 7, 4, 9, 1, 6, 2, 8};
        int[] highs = {6, 5, 4, 9, 4, 12, 1, 6, 2, 8};
        int previousEnd = 0;
        int previousComparisons = 0;
        for (int i = 0; i < lows.length; i++) {
            int got = column.rangeCount(lows[i], highs[i]);
            check(got == bruteRangeCount(base, lows[i], highs[i]),
                    "probe [" + lows[i] + "," + highs[i] + "] -> " + got);
            check(column.verifyPartition() == null, "invariant after probe " + i);
            check(column.crackEnd >= previousEnd, "crackEnd monotone at probe " + i);
            previousEnd = column.crackEnd;
            check(column.comparisons >= previousComparisons, "comparisons monotone");
            previousComparisons = (int) column.comparisons;
        }
        check(column.crackEnd == base.length, "all probes together cracked the whole column");
        check(column.cracks == 11, "eleven distinct pivots opened, got " + column.cracks);

        // --- cached pivot: a repeated probe is free ---------------------------
        long before = column.comparisons;
        int again = column.rangeCount(3, 6);
        check(again == bruteRangeCount(base, 3, 6), "repeated probe result");
        check(column.comparisons == before, "repeated probe cost 0 comparisons");
        int repeatedAgain = column.rangeCount(3, 6);
        check(repeatedAgain == again, "repeated probe stable");

        // --- emergent property: cracking every distinct value fully sorts ------
        int[] values = randomColumn(96, 24);
        CrackedColumn sorter = new CrackedColumn(values);
        boolean[] seen = new boolean[24];
        List<Integer> distinct = new ArrayList<>();
        for (int value : values) if (!seen[value]) { seen[value] = true; distinct.add(value); }
        Collections.sort(distinct);
        for (int value : distinct) sorter.rangeCount(value, value);
        check(sorter.verifyPartition() == null, "sorted column invariant");
        int[] sortedCopy = values.clone();
        Arrays.sort(sortedCopy);
        check(Arrays.equals(sorter.data, sortedCopy), "cracking at every value sorts the column");

        // --- adaptive workload beats rescan-per-query -------------------------
        int[] big = randomColumn(1024, 4096);
        CrackedColumn adaptive = new CrackedColumn(big);
        int[] probes = {500, 1500, 2500, 750, 2000, 3000};
        long naiveComparisons = 0;
        long firstRoundComparisons = 0;
        for (int round = 0; round < 6; round++) {
            for (int probe : probes) {
                check(adaptive.rangeCount(probe, probe + 300) == bruteRangeCount(big, probe, probe + 300),
                        "adaptive probe " + probe + " round " + round);
                naiveComparisons += big.length;
            }
            check(adaptive.verifyPartition() == null, "adaptive invariant after round " + round);
            if (round == 0) {
                firstRoundComparisons = adaptive.comparisons;
            } else {
                // every probe of a later round hits pivots that are already cracked
                check(adaptive.comparisons == firstRoundComparisons,
                        "round " + round + " must be comparison free");
            }
        }
        check(adaptive.crackEnd == big.length, "adaptive cracked the whole column");
        check(adaptive.cracks == 2 * probes.length, "one crack per distinct bound, got " + adaptive.cracks);
        check(adaptive.comparisons * 4 < naiveComparisons,
                "cracking work " + adaptive.comparisons + " should be far below rescan " + naiveComparisons);
        check(adaptive.comparisons > 0, "cracking really did work");
        check(adaptive.comparisons <= 12L * big.length, "each of the twelve cracks scans at most the column");

        // --- invariant stress on pseudo random probes -------------------------
        int[] columnData = randomColumn(200, 40);
        CrackedColumn stress = new CrackedColumn(columnData);
        for (int i = 0; i < 120; i++) {
            int lo = nextInt(45);
            int hi = lo + nextInt(10);
            int got = stress.rangeCount(lo, hi);
            check(got == bruteRangeCount(columnData, lo, hi), "stress probe " + i + " [" + lo + "," + hi + "]");
            check(stress.verifyPartition() == null, "stress invariant " + i);
        }
        check(stress.crackEnd == columnData.length, "stress cracked whole column");
        check(stress.rangeCount(-5, 100) == columnData.length, "full range count");
        check(stress.rangeCount(1000, 2000) == 0, "range outside domain");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
