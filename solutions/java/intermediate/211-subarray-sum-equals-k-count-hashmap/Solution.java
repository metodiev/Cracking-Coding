/** Task 211 - Subarray sum equals K count (hashmap). (Tier: Intermediate, CTCI Ch 16) */
import java.util.*;

public class Solution {
    static int countSubarraysSumK(int[] nums, int target) {
        Map<Integer, Integer> counts = new HashMap<>();
        counts.put(0, 1);
        int prefix = 0, total = 0;
        for (int x : nums) {
            prefix += x;
            total += counts.getOrDefault(prefix - target, 0);
            counts.merge(prefix, 1, Integer::sum);
        }
        return total;
    }

    public static void main(String[] args) {
        check(countSubarraysSumK(new int[]{1, 1, 1}, 2) == 2, "ones");
        check(countSubarraysSumK(new int[]{1, 2, 3}, 3) == 2, "two");
        check(countSubarraysSumK(new int[]{-1, -1, 1}, 0) == 1, "mixed");
        check(countSubarraysSumK(new int[]{1, 2, 3}, 100) == 0, "none");
        check(countSubarraysSumK(new int[]{0, 0}, 0) == 3, "zeros");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
