/** Task 086 - Count inversion pairs using naive O(n^2). (Tier: Beginner, CTCI Ch 10)
 *
 * An inversion is a pair i < j with a[i] > a[j]; equal values are not inversions.
 */
public class Solution {
    static int countInversionsNaive(int[] nums) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] > nums[j]) count++;
            }
        }
        return count;
    }

    public static void main(String[] args) {
        check(countInversionsNaive(new int[]{}) == 0, "empty");
        check(countInversionsNaive(new int[]{1}) == 0, "single");
        check(countInversionsNaive(new int[]{1, 2, 3}) == 0, "sorted");
        check(countInversionsNaive(new int[]{1, 1, 1}) == 0, "all equal");
        check(countInversionsNaive(new int[]{2, 2}) == 0, "pair of equals");
        check(countInversionsNaive(new int[]{1, 3, 2}) == 1, "one inversion");
        check(countInversionsNaive(new int[]{3, 2, 1}) == 3, "reverse 3");
        check(countInversionsNaive(new int[]{2, 4, 1, 3, 5}) == 3, "mixed");
        check(countInversionsNaive(new int[]{-1, -2, -3}) == 3, "negatives");
        check(countInversionsNaive(new int[]{5, 4, 3, 2, 1}) == 10, "reverse 5");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
