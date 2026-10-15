/** Task 204 - Count pairs with given difference K in array (hash). (Tier: Intermediate, CTCI Ch 16) */
import java.util.*;

public class Solution {
    static long countPairsWithDifference(int[] nums, int k) {
        if (k < 0) throw new IllegalArgumentException("k must be >= 0");
        Map<Integer, Integer> freq = new HashMap<>();
        for (int x : nums) freq.merge(x, 1, Integer::sum);
        long total = 0;
        if (k == 0) {
            for (int c : freq.values()) total += (long) c * (c - 1) / 2;
            return total;
        }
        for (Map.Entry<Integer, Integer> e : freq.entrySet())
            total += (long) e.getValue() * freq.getOrDefault(e.getKey() + k, 0);
        return total;
    }

    public static void main(String[] args) {
        check(countPairsWithDifference(new int[]{1, 5, 3, 4, 2}, 2) == 3, "classic");
        check(countPairsWithDifference(new int[]{1, 1, 1, 1}, 0) == 6, "k=0 combos");
        check(countPairsWithDifference(new int[]{1, 2, 3, 4}, 10) == 0, "none");
        check(countPairsWithDifference(new int[]{}, 3) == 0, "empty");
        check(countPairsWithDifference(new int[]{5, 5, 5, 1}, 4) == 3, "multiplicity");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
