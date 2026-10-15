/** Task 074 - Implement simple stable partition. (Tier: Beginner, CTCI Ch 10) */
import java.util.Arrays;

public class Solution {
    static int[] stablePartition(int[] items, java.util.function.IntPredicate predicate) {
        int[] out = new int[items.length];
        int k = 0;
        for (int x : items) if (predicate.test(x)) out[k++] = x;
        for (int x : items) if (!predicate.test(x)) out[k++] = x;
        return out;
    }

    public static void main(String[] args) {
        check(Arrays.equals(stablePartition(new int[]{1, 2, 3, 4, 5, 6}, x -> x % 2 == 0),
                new int[]{2, 4, 6, 1, 3, 5}), "evens first");
        check(stablePartition(new int[]{}, x -> true).length == 0, "empty");
        check(Arrays.equals(stablePartition(new int[]{4, 1, 4, 2}, x -> x == 4),
                new int[]{4, 4, 1, 2}), "stability");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
