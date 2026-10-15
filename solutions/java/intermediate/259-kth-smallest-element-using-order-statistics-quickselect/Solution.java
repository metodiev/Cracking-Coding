/** Task 259 - Kth smallest element using order statistics / quickselect. (Tier: Intermediate, CTCI Beyond — Numerical & Scientific Computing) */
import java.util.*;

public class Solution {
    static final long LCG_MOD = 2147483647L;
    static final long LCG_MUL = 48271L;

    /** Lomuto partition with a[hi] as the deterministic pivot; returns its final index. */
    static int partition(int[] a, int lo, int hi) {
        int pivot = a[hi];
        int store = lo;
        for (int i = lo; i < hi; i++) {
            if (a[i] <= pivot) {
                int tmp = a[store];
                a[store] = a[i];
                a[i] = tmp;
                store++;
            }
        }
        int tmp = a[store];
        a[store] = a[hi];
        a[hi] = tmp;
        return store;
    }

    /** The k-th smallest value (1-based k); the caller's array is copied first. */
    static int kthSmallest(int[] values, int k) {
        if (k < 1 || k > values.length) throw new IllegalArgumentException("k out of range");
        int[] a = Arrays.copyOf(values, values.length);
        int lo = 0;
        int hi = a.length - 1;
        int target = k - 1;
        while (true) {
            int p = partition(a, lo, hi);          // iterative: a bad pivot never grows a stack
            if (p == target) return a[p];
            if (p < target) lo = p + 1;
            else hi = p - 1;
        }
    }

    /** Park-Miller minimal standard generator: identical values in every language. */
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
        check(kthSmallest(new int[]{3, 1, 2}, 1) == 1, "k=1");
        check(kthSmallest(new int[]{3, 1, 2}, 2) == 2, "k=2");
        check(kthSmallest(new int[]{3, 1, 2}, 3) == 3, "k=3");
        check(kthSmallest(new int[]{5, 5, 5, 5}, 3) == 5, "all equal");
        check(kthSmallest(new int[]{2, 1, 1, 3}, 2) == 1, "duplicates: sorted [1,1,2,3]");

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

        // Fixed cases (duplicates included) cross-checked against the sorted() oracle, all k.
        List<int[]> cases = new ArrayList<>();
        cases.add(lcgValues(259001, 1, 5));
        cases.add(lcgValues(259002, 2, 5));
        cases.add(lcgValues(259003, 3, 7));
        cases.add(lcgValues(259004, 5, 20));
        cases.add(lcgValues(259005, 8, 3));
        cases.add(lcgValues(259006, 16, 1000));
        cases.add(lcgValues(259007, 33, 10));
        cases.add(lcgValues(259008, 50, 50));
        cases.add(range(64));
        cases.add(reverseRange(64));
        cases.add(repeat(7, 32));
        cases.add(new int[]{5, 5, 4, 4, 3, 3, 2, 2, 1, 1});
        cases.add(lcgValues(259009, 200, 10000));
        for (int[] c : cases) check(caseAgainstOracle(c), "fixed case vs sorted oracle");
        System.out.println("PASS");
    }

    static boolean caseAgainstOracle(int[] values) {
        int[] ordered = Arrays.copyOf(values, values.length);
        Arrays.sort(ordered);
        for (int k = 1; k <= values.length; k++) {
            if (kthSmallest(values, k) != ordered[k - 1]) return false;
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

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
