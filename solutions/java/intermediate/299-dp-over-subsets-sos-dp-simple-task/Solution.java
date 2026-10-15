/** Task 299 - DP over subsets (SOS DP simple task). (Tier: Intermediate, CTCI Beyond — Networking, OS & Real-Time) */
import java.util.Arrays;

public class Solution {
    /** f[mask] = sum of a[sub] for all sub ⊆ mask. Naive work is 3^n; this is n * 2^n. */
    static int[] sumOverSubsets(int[] a) {
        int n = Integer.numberOfTrailingZeros(Integer.highestOneBit(a.length));
        if (a.length != (1 << n) || a.length == 0) throw new IllegalArgumentException("length must be a power of two");
        int[] f = a.clone();
        for (int i = 0; i < n; i++) {
            int bit = 1 << i;
            for (int mask = 0; mask < f.length; mask++) {
                if ((mask & bit) != 0) f[mask] += f[mask ^ bit];
            }
        }
        return f;
    }

    /** Oracle: enumerate every submask of every mask explicitly (the O(3^n) loop). */
    static int[] bruteSos(int[] a) {
        int[] result = new int[a.length];
        for (int mask = 0; mask < a.length; mask++) {
            int total = 0;
            int sub = mask;
            while (true) {
                total += a[sub];
                if (sub == 0) break;
                sub = (sub - 1) & mask;
            }
            result[mask] = total;
        }
        return result;
    }

    private static int seed = 606;

    private static int[] lcgArray(int size) {
        int[] values = new int[size];
        for (int i = 0; i < size; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % 50;
        }
        return values;
    }

    public static void main(String[] args) {
        check(Arrays.equals(sumOverSubsets(new int[]{1, 2, 4, 8}), new int[]{1, 3, 5, 15}), "n = 2");
        check(Arrays.equals(sumOverSubsets(new int[]{7}), new int[]{7}), "n = 0");

        for (int n : new int[]{1, 2, 3, 4, 10}) {
            int[] a = lcgArray(1 << n);
            int[] f = sumOverSubsets(a);
            check(Arrays.equals(f, bruteSos(a)), "n = " + n + " vs brute force");
            long total = 0;
            for (int mask = 0; mask < a.length; mask++) {
                total += (long) a[mask] << (n - Integer.bitCount(mask));
            }
            long sumF = 0;
            for (int value : f) sumF += value;
            check(sumF == total, "n = " + n + " superset identity");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
