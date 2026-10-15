/** Task 073 - Radix sort for non-negative integers. (Tier: Beginner, CTCI Ch 10) */
import java.util.Arrays;

public class Solution {
    static int[] radixSort(int[] a) {
        if (a.length == 0) return new int[0];
        int max = 0;
        for (int x : a) max = Math.max(max, x);
        int[] out = a.clone();
        for (long exp = 1; max / exp > 0; exp *= 10) {
            int[] buckets = new int[10];
            for (int x : out) buckets[(int) ((x / exp) % 10)]++;
            for (int i = 1; i < 10; i++) buckets[i] += buckets[i - 1];
            int[] next = new int[out.length];
            for (int i = out.length - 1; i >= 0; i--) {
                int digit = (int) ((out[i] / exp) % 10);
                next[--buckets[digit]] = out[i];
            }
            out = next;
        }
        return out;
    }

    public static void main(String[] args) {
        check(Arrays.equals(radixSort(new int[]{170, 45, 75, 90, 802, 24, 2, 66}),
                new int[]{2, 24, 45, 66, 75, 90, 170, 802}), "classic");
        check(radixSort(new int[]{}).length == 0, "empty");
        check(Arrays.equals(radixSort(new int[]{0, 10, 100}), new int[]{0, 10, 100}), "powers of ten");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
