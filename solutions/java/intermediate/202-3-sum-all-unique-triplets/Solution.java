/** Task 202 - 3-sum all unique triplets. (Tier: Intermediate, CTCI Ch 16) */
import java.util.*;

public class Solution {
    static List<List<Integer>> threeSum(int[] nums) {
        int[] a = nums.clone();
        Arrays.sort(a);
        List<List<Integer>> out = new ArrayList<>();
        for (int i = 0; i + 2 < a.length; i++) {
            if (i > 0 && a[i] == a[i - 1]) continue;
            if (a[i] > 0) break;
            int lo = i + 1, hi = a.length - 1;
            while (lo < hi) {
                int total = a[i] + a[lo] + a[hi];
                if (total == 0) {
                    out.add(List.of(a[i], a[lo], a[hi]));
                    while (lo < hi && a[lo] == a[lo + 1]) lo++;
                    while (lo < hi && a[hi] == a[hi - 1]) hi--;
                    lo++;
                    hi--;
                } else if (total < 0) {
                    lo++;
                } else {
                    hi--;
                }
            }
        }
        return out;
    }

    public static void main(String[] args) {
        check(threeSum(new int[]{-1, 0, 1, 2, -1, -4}).toString().equals("[[-1, -1, 2], [-1, 0, 1]]"), "classic");
        check(threeSum(new int[]{0, 1, 1}).isEmpty(), "no triplet");
        check(threeSum(new int[]{0, 0, 0}).toString().equals("[[0, 0, 0]]"), "triple zeros");
        check(threeSum(new int[]{-2, 0, 0, 2, 2}).toString().equals("[[-2, 0, 2]]"), "duplicate dedup");
        check(threeSum(new int[]{}).isEmpty(), "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
