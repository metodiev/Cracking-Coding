/** Task 292 - Minimum subset sum difference (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** Returns {difference, side...}: minimum |sum(A) - sum(B)| and one achieving subset. */
    static int[] minDifference(int[] nums) {
        int total = 0;
        for (int value : nums) total += value;
        int half = total / 2;
        int n = nums.length;
        boolean[][] reachable = new boolean[n + 1][half + 1];
        reachable[0][0] = true;
        for (int i = 1; i <= n; i++) {
            int value = nums[i - 1];
            for (int s = 0; s <= half; s++) {
                reachable[i][s] = reachable[i - 1][s] || (s >= value && reachable[i - 1][s - value]);
            }
        }
        int best = half;
        while (best > 0 && !reachable[n][best]) best--;
        int[] reversed = new int[n];
        int count = 0;
        int s = best;
        for (int i = n; i >= 1; i--) {
            int value = nums[i - 1];
            if (s >= value && reachable[i - 1][s - value]) {
                reversed[count++] = value;
                s -= value;
            }
        }
        int[] result = new int[count + 1];
        result[0] = total - 2 * best;
        for (int i = 0; i < count; i++) result[i + 1] = reversed[count - 1 - i];
        return result;
    }

    static int bruteMinDifference(int[] nums) {
        int total = 0;
        for (int value : nums) total += value;
        int n = nums.length;
        int best = total;
        for (int mask = 0; mask < (1 << n); mask++) {
            int s = 0;
            for (int i = 0; i < n; i++) if (((mask >> i) & 1) != 0) s += nums[i];
            best = Math.min(best, Math.abs(total - 2 * s));
        }
        return best;
    }

    private static int seed = 777;

    private static int[] lcgNumbers(int n) {
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            nums[i] = 1 + seed % 25;
        }
        return nums;
    }

    public static void main(String[] args) {
        int[] result = minDifference(new int[]{1, 6, 11, 5});
        check(result[0] == 1 && result.length == 3, "1,6,11,5 difference");
        int sideSum = 0;
        for (int i = 1; i < result.length; i++) sideSum += result[i];
        check(sideSum == 11, "1,6,11,5 side sum");

        result = minDifference(new int[]{3, 3, 3});
        check(result[0] == 3 && result.length == 2 && result[1] == 3, "3,3,3");

        result = minDifference(new int[]{1, 2, 3, 4});
        check(result[0] == 0 && result.length == 3 && result[1] + result[2] == 5, "1,2,3,4");

        result = minDifference(new int[]{7});
        check(result[0] == 7 && result.length == 1, "single element");

        result = minDifference(new int[]{});
        check(result[0] == 0 && result.length == 1, "empty");

        for (int trial = 0; trial < 200; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 14;
            int[] nums = lcgNumbers(count);
            int[] answer = minDifference(nums);
            int total = 0;
            for (int value : nums) total += value;
            int chosen = 0;
            for (int i = 1; i < answer.length; i++) chosen += answer[i];
            check(answer[0] == bruteMinDifference(nums), "random trial " + trial);
            check(total - 2 * chosen == answer[0], "random reconstruction " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
