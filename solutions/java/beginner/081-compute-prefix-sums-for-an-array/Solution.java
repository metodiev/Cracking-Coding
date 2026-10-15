/** Task 081 - Compute prefix sums for an array. (Tier: Beginner, CTCI Ch 1)
 *
 * prefix[i] = sum of the first i elements, so prefix.length == a.length + 1.
 */
import java.util.*;

public class Solution {
    static int[] prefixSums(int[] a) {
        int[] prefix = new int[a.length + 1];
        for (int i = 0; i < a.length; i++) {
            prefix[i + 1] = prefix[i] + a[i];
        }
        return prefix;
    }

    public static void main(String[] args) {
        check(Arrays.equals(prefixSums(new int[]{}), new int[]{0}), "empty");
        check(Arrays.equals(prefixSums(new int[]{5}), new int[]{0, 5}), "single");
        check(Arrays.equals(prefixSums(new int[]{1, 2, 3}), new int[]{0, 1, 3, 6}), "1,2,3");
        check(Arrays.equals(prefixSums(new int[]{-1, 1, -1}), new int[]{0, -1, 0, -1}), "negatives");
        check(Arrays.equals(prefixSums(new int[]{4, 4, 4, 4}), new int[]{0, 4, 8, 12, 16}), "duplicates");
        check(prefixSums(new int[]{1, 2, 3, 4, 5}).length == 6, "length n+1");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
