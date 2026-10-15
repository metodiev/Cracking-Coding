/** Task 004 - Remove duplicates from an array (create new array). (Tier: Beginner, CTCI Ch 1) */
import java.util.*;

public class Solution {
    static int[] removeDuplicates(int[] items) {
        Set<Integer> seen = new LinkedHashSet<>();
        for (int x : items) seen.add(x);
        int[] out = new int[seen.size()];
        int i = 0;
        for (int x : seen) out[i++] = x;
        return out;
    }

    public static void main(String[] args) {
        check(Arrays.equals(removeDuplicates(new int[]{1, 2, 2, 3, 1}), new int[]{1, 2, 3}), "basic");
        check(removeDuplicates(new int[]{}).length == 0, "empty");
        check(Arrays.equals(removeDuplicates(new int[]{5, 5, 5}), new int[]{5}), "all same");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
