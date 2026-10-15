/** Task 216 - Partition array into k equal-sum subsets (backtracking). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static boolean canPartitionK(int[] nums, int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be positive");
        int total = 0;
        for (int x : nums) total += x;
        if (total % k != 0) return false;
        int target = total / k;
        int[] a = nums.clone();
        Arrays.sort(a);
        for (int i = 0, j = a.length - 1; i < j; i++, j--) { int t = a[i]; a[i] = a[j]; a[j] = t; }
        if (a.length > 0 && a[0] > target) return false;
        return place(a, 0, new int[k], target);
    }

    private static boolean place(int[] nums, int i, int[] buckets, int target) {
        if (i == nums.length) return true;
        Set<Integer> seen = new HashSet<>();
        for (int b = 0; b < buckets.length; b++) {
            if (seen.contains(buckets[b])) continue;
            if (buckets[b] + nums[i] <= target) {
                seen.add(buckets[b]);
                buckets[b] += nums[i];
                if (place(nums, i + 1, buckets, target)) return true;
                buckets[b] -= nums[i];
            }
            if (buckets[b] == 0) break;
        }
        return false;
    }

    public static void main(String[] args) {
        check(canPartitionK(new int[]{4, 3, 2, 3, 5, 2, 1}, 4), "classic");
        check(canPartitionK(new int[]{1, 2, 3, 4}, 2), "two halves");
        check(canPartitionK(new int[]{1, 1, 1, 1}, 2), "four ones");
        check(!canPartitionK(new int[]{2, 2, 2, 2, 3}, 3), "not divisible");
        check(!canPartitionK(new int[]{2, 2, 2, 2, 2, 2}, 4), "twos cannot make 3s");
        check(canPartitionK(new int[]{10, 10, 10, 10, 10, 10}, 3), "three tens");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
