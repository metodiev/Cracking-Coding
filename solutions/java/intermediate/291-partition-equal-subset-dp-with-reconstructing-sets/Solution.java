/** Task 291 - Partition equal subset (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8) */
import java.util.Arrays;

public class Solution {
    /** Returns one equal-sum side when nums splits in half, else null. */
    static int[] partition(int[] nums) {
        int total = 0;
        for (int value : nums) total += value;
        if (total % 2 != 0) return null;
        int target = total / 2;
        int n = nums.length;
        boolean[][] reachable = new boolean[n + 1][target + 1];
        reachable[0][0] = true;
        for (int i = 1; i <= n; i++) {
            int value = nums[i - 1];
            for (int s = 0; s <= target; s++) {
                reachable[i][s] = reachable[i - 1][s] || (s >= value && reachable[i - 1][s - value]);
            }
        }
        if (!reachable[n][target]) return null;
        int[] reversed = new int[n];
        int count = 0;
        int s = target;
        for (int i = n; i >= 1; i--) {
            int value = nums[i - 1];
            if (s >= value && reachable[i - 1][s - value]) {
                reversed[count++] = value;
                s -= value;
            }
        }
        int[] side = new int[count];
        for (int i = 0; i < count; i++) side[i] = reversed[count - 1 - i];
        return side;
    }

    /** True when side is a sub-multiset of nums whose complement has the same sum. */
    static boolean validate(int[] nums, int[] side) {
        boolean[] used = new boolean[nums.length];
        int sideSum = 0;
        for (int value : side) {
            sideSum += value;
            boolean found = false;
            for (int i = 0; i < nums.length && !found; i++) {
                if (!used[i] && nums[i] == value) {
                    used[i] = true;
                    found = true;
                }
            }
            if (!found) return false;
        }
        int restSum = 0;
        for (int i = 0; i < nums.length; i++) if (!used[i]) restSum += nums[i];
        return sideSum == restSum;
    }

    static boolean brutePartitionExists(int[] nums) {
        int total = 0;
        for (int value : nums) total += value;
        if (total % 2 != 0) return false;
        int target = total / 2;
        int n = nums.length;
        for (int mask = 0; mask < (1 << n); mask++) {
            int sum = 0;
            for (int i = 0; i < n; i++) if (((mask >> i) & 1) != 0) sum += nums[i];
            if (sum == target) return true;
        }
        return false;
    }

    private static int seed = 1234;

    private static int[] lcgNumbers(int n) {
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            nums[i] = 1 + seed % 15;
        }
        return nums;
    }

    public static void main(String[] args) {
        int[] sample = new int[]{1, 5, 11, 5};
        int[] side = partition(sample);
        check(side != null && Arrays.equals(side, new int[]{1, 5, 5}), "sample split");
        check(validate(sample, side), "sample validation");

        check(partition(new int[]{1, 2, 3, 5}) == null, "odd total");
        check(partition(new int[]{1, 2}) == null, "odd total small");
        check(partition(new int[]{}).length == 0, "empty set");
        check(partition(new int[]{2, 2, 2}) == null, "even total but unreachable half");

        int[] four = new int[]{1, 2, 3, 4};
        side = partition(four);
        int sideSum = 0;
        for (int value : side) sideSum += value;
        check(sideSum == 5 && validate(four, side), "1..4 split");

        for (int trial = 0; trial < 200; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 2 + seed % 8;
            int[] nums = lcgNumbers(count);
            int[] randomSide = partition(nums);
            check((randomSide != null) == brutePartitionExists(nums), "random trial " + trial);
            if (randomSide != null) check(validate(nums, randomSide), "random validation " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
