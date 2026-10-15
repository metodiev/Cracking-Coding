/** Task 085 - Three-sum (return any triplet). (Tier: Beginner, CTCI Ch 1)
 *
 * Sort a copy, then fix the smallest value and use two pointers for the rest:
 * O(n^2) time. Returns any triplet of values (v1 <= v2 <= v3) drawn from three
 * distinct positions that sums to 0, or an empty array when none exists.
 */
import java.util.*;

public class Solution {
    static int[] threeSum(int[] nums) {
        if (nums.length < 3) return new int[0];
        int[] values = Arrays.copyOf(nums, nums.length);
        Arrays.sort(values);
        int n = values.length;
        for (int i = 0; i < n - 2; i++) {
            if (i > 0 && values[i] == values[i - 1]) continue;
            int lo = i + 1, hi = n - 1;
            while (lo < hi) {
                int total = values[i] + values[lo] + values[hi];
                if (total == 0) return new int[]{values[i], values[lo], values[hi]};
                if (total < 0) lo++;
                else hi--;
            }
        }
        return new int[0];
    }

    static boolean isSorted(int[] a) {
        for (int i = 1; i < a.length; i++) if (a[i - 1] > a[i]) return false;
        return true;
    }

    static boolean containsTriplet(int[] nums, int[] triplet) {
        if (triplet.length != 3) return false;
        boolean[] used = new boolean[nums.length];
        for (int v : triplet) {
            boolean found = false;
            for (int i = 0; i < nums.length; i++) {
                if (!used[i] && nums[i] == v) {
                    used[i] = true;
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        int[] nums = {-1, 0, 1, 2, -1, -4};
        int[] t = threeSum(nums);
        check(t.length == 3, "triplet found");
        check(t[0] + t[1] + t[2] == 0, "sums to zero");
        check(isSorted(t), "sorted output");
        check(containsTriplet(nums, t), "three distinct positions");

        check(Arrays.equals(threeSum(new int[]{0, 0, 0}), new int[]{0, 0, 0}), "0,0,0");
        check(Arrays.equals(threeSum(new int[]{-3, 1, 2}), new int[]{-3, 1, 2}), "exact triplet");
        check(Arrays.equals(threeSum(new int[]{1, -1, 0}), new int[]{-1, 0, 1}), "unsorted input");
        check(Arrays.equals(threeSum(new int[]{-2, 0, 0, 2, 2}), new int[]{-2, 0, 2}), "duplicates");
        check(threeSum(new int[]{1, 2, 3}).length == 0, "none");
        check(threeSum(new int[]{1, 1, 1}).length == 0, "no zero sum");
        check(threeSum(new int[]{0, 0}).length == 0, "too short");
        check(threeSum(new int[]{}).length == 0, "empty");

        int[] nums2 = {8, -4, -2, 3, 1, -5, 7};
        int[] t2 = threeSum(nums2);
        check(t2.length == 3 && t2[0] + t2[1] + t2[2] == 0, "arbitrary order: sums to zero");
        check(isSorted(t2) && containsTriplet(nums2, t2), "arbitrary order: valid triplet");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
