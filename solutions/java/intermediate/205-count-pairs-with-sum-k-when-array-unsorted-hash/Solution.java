/** Task 205 - Count pairs with sum K when array unsorted (hash). (Tier: Intermediate, CTCI Ch 16) */
import java.util.*;

public class Solution {
    static long countPairsWithSum(int[] nums, int target) {
        Map<Integer, Integer> seen = new HashMap<>();
        long total = 0;
        for (int x : nums) {
            total += seen.getOrDefault(target - x, 0);
            seen.merge(x, 1, Integer::sum);
        }
        return total;
    }

    public static void main(String[] args) {
        check(countPairsWithSum(new int[]{1, 5, 3, 3, 3}, 6) == 4, "mixed");
        check(countPairsWithSum(new int[]{1, 2, 3, 4}, 5) == 2, "two pairs");
        check(countPairsWithSum(new int[]{1, 1, 1, 1}, 2) == 6, "all ones");
        check(countPairsWithSum(new int[]{1, 2, 3}, 100) == 0, "none");
        check(countPairsWithSum(new int[]{}, 5) == 0, "empty");
        check(countPairsWithSum(new int[]{0, 0}, 0) == 1, "zeros");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
