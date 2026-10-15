/** Task 207 - Three-sum closest. (Tier: Intermediate, CTCI Ch 16) */
import java.util.*;

public class Solution {
    static int threeSumClosest(int[] nums, int target) {
        if (nums.length < 3) throw new IllegalArgumentException("need at least three values");
        int[] a = nums.clone();
        Arrays.sort(a);
        int best = a[0] + a[1] + a[2];
        for (int i = 0; i + 2 < a.length; i++) {
            int lo = i + 1, hi = a.length - 1;
            while (lo < hi) {
                int total = a[i] + a[lo] + a[hi];
                if (Math.abs(total - target) < Math.abs(best - target)) best = total;
                if (total < target) lo++;
                else if (total > target) hi--;
                else return total;
            }
        }
        return best;
    }

    public static void main(String[] args) {
        check(threeSumClosest(new int[]{-1, 2, 1, -4}, 1) == 2, "classic");
        check(threeSumClosest(new int[]{0, 0, 0}, 1) == 0, "zeros");
        check(threeSumClosest(new int[]{1, 1, 1, 1}, 5) == 3, "all ones");
        check(threeSumClosest(new int[]{-5, -2, -1}, -3) == -8, "negatives");

        boolean threw = false;
        try { threeSumClosest(new int[]{1, 2}, 3); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "needs three values");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
