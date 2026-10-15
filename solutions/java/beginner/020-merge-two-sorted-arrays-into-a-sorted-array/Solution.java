/** Task 020 - Merge two sorted arrays into a sorted array. (Tier: Beginner, CTCI Ch 10) */
import java.util.Arrays;

public class Solution {
    static int[] mergeSorted(int[] a, int[] b) {
        int[] out = new int[a.length + b.length];
        int i = 0, j = 0, k = 0;
        while (i < a.length && j < b.length)
            out[k++] = a[i] <= b[j] ? a[i++] : b[j++];
        while (i < a.length) out[k++] = a[i++];
        while (j < b.length) out[k++] = b[j++];
        return out;
    }

    public static void main(String[] args) {
        check(Arrays.equals(mergeSorted(new int[]{1, 3, 5}, new int[]{2, 4, 6}),
                new int[]{1, 2, 3, 4, 5, 6}), "basic");
        check(Arrays.equals(mergeSorted(new int[]{}, new int[]{1, 2}), new int[]{1, 2}), "empty first");
        check(Arrays.equals(mergeSorted(new int[]{2, 2}, new int[]{2}), new int[]{2, 2, 2}), "duplicates");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
