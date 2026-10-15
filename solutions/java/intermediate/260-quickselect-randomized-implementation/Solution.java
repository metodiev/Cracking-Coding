/** Task 260 - Quickselect randomized implementation. (Tier: Intermediate, CTCI Ch 10) */
import java.util.*;

public class Solution {
    static final long LCG_MOD = 2147483647L;
    static final long LCG_MUL = 48271L;
    static final long SEED = 260260L;

    /** Dutch flag; returns {lt, gt}: a[lo..lt-1] < pivot == a[lt..gt], a[gt+1..hi] > pivot. */
    static int[] partition3(int[] a, int lo, int hi, int pivot) {
        int lt = lo;
        int i = lo;
        int gt = hi;
        while (i <= gt) {
            if (a[i] < pivot) {
                int tmp = a[lt];
                a[lt] = a[i];
                a[i] = tmp;
                lt++;
                i++;
            } else if (a[i] > pivot) {
                int tmp = a[i];
                a[i] = a[gt];
                a[gt] = tmp;
                gt--;
            } else {
                i++;
            }
        }
        return new int[]{lt, gt};
    }

    /**
     * Randomized quickselect with a fixed-seed LCG (reproducible runs); the returned value
     * never depends on the seed. Duplicates collapse into one equal block, so all-equal
     * input finishes after a single partition; the loop is iterative, so no stack blow-up.
     * Expected O(n), worst case O(n^2). k is 1-based; the caller's array is copied.
     */
    static int kthSmallest(int[] values, int k, long seed) {
        if (k < 1 || k > values.length) throw new IllegalArgumentException("k out of range");
        int[] a = Arrays.copyOf(values, values.length);
        long state = seed % LCG_MOD;
        if (state == 0) state = 1;
        int lo = 0;
        int hi = a.length - 1;
        int target = k - 1;
        while (true) {
            state = (state * LCG_MUL) % LCG_MOD;
            int pivot = a[(int) (lo + state % (hi - lo + 1))];
            int[] block = partition3(a, lo, hi, pivot);
            if (target < block[0]) hi = block[0] - 1;
            else if (target > block[1]) lo = block[1] + 1;
            else return a[target];
        }
    }

    static int kthSmallest(int[] values, int k) {
        return kthSmallest(values, k, SEED);
    }

    static int[] lcgValues(long seed, int count, int bound) {
        int[] out = new int[count];
        long state = seed;
        for (int i = 0; i < count; i++) {
            state = (state * LCG_MUL) % LCG_MOD;
            out[i] = (int) (state % bound);
        }
        return out;
    }

    public static void main(String[] args) {
        check(kthSmallest(new int[]{7}, 1) == 7, "single element");
        check(kthSmallest(new int[]{3, 1, 2}, 2) == 2, "k=2");
        check(kthSmallest(new int[]{5, 5, 5, 5}, 3) == 5, "all equal small");
        check(kthSmallest(new int[]{2, 1, 1, 3}, 2) == 1, "duplicates");

        int[] original = {9, 4, 7, 1, 8, 2, 6, 3, 5};
        check(kthSmallest(original, 4) == 4, "middle");
        check(Arrays.equals(original, new int[]{9, 4, 7, 1, 8, 2, 6, 3, 5}), "input untouched");

        for (int bad : new int[]{0, -1, 10}) {
            boolean threw = false;
            try {
                kthSmallest(new int[]{1, 2, 3}, bad);
            } catch (IllegalArgumentException e) {
                threw = true;
            }
            check(threw, "k out of range throws: " + bad);
        }

        List<int[]> cases = new ArrayList<>();
        cases.add(lcgValues(260001, 1, 5));
        cases.add(lcgValues(260002, 2, 5));
        cases.add(lcgValues(260003, 3, 7));
        cases.add(lcgValues(260004, 5, 20));
        cases.add(lcgValues(260005, 8, 3));
        cases.add(lcgValues(260006, 16, 1000));
        cases.add(lcgValues(260007, 33, 10));
        cases.add(lcgValues(260008, 50, 50));
        cases.add(range(64));
        cases.add(reverseRange(64));
        cases.add(repeat(9, 17));
        cases.add(alternating(200));
        cases.add(lcgValues(260009, 200, 10000));
        for (int[] c : cases) check(agreesWithOracle(c), "oracle agreement");

        int[] bigEqual = repeat(7, 5000);
        for (int k : new int[]{1, 2, 2500, 4999, 5000}) check(kthSmallest(bigEqual, k) == 7, "big all-equal k=" + k);
        int[] ascending = range(4000);
        int[] descending = reverseRange(4000);
        int[] sawtooth = sawtooth(1000);
        int[] sawOrdered = Arrays.copyOf(sawtooth, sawtooth.length);
        Arrays.sort(sawOrdered);
        for (int k : new int[]{1, 2000, 4000}) {
            check(kthSmallest(ascending, k) == k - 1, "ascending k=" + k);
            check(kthSmallest(descending, k) == k - 1, "descending k=" + k);
        }
        for (int k : new int[]{1, 500, 1000}) {
            check(kthSmallest(sawtooth, k) == sawOrdered[k - 1], "sawtooth k=" + k);
        }
        System.out.println("PASS");
    }

    /** Every k agrees with sorted order, and the value is seed-independent. */
    static boolean agreesWithOracle(int[] values) {
        int[] ordered = Arrays.copyOf(values, values.length);
        Arrays.sort(ordered);
        for (int k = 1; k <= values.length; k++) {
            if (kthSmallest(values, k) != ordered[k - 1]) return false;
            if (kthSmallest(values, k, 999) != ordered[k - 1]) return false;
        }
        return true;
    }

    static int[] range(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = i;
        return out;
    }

    static int[] reverseRange(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = n - 1 - i;
        return out;
    }

    static int[] repeat(int value, int n) {
        int[] out = new int[n];
        Arrays.fill(out, value);
        return out;
    }

    static int[] alternating(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = 1 + i % 2;
        return out;
    }

    static int[] sawtooth(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = i % 17;
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
