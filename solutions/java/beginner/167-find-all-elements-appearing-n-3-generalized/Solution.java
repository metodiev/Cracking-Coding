/** Task 167 - Find all elements appearing > n/3 (generalized). (Tier: Beginner, CTCI Ch 6) */
import java.util.*;

public class Solution {
    /** Returns a sorted list of values occurring more than n/3 times. */
    static List<Integer> majorityElementsN3(int[] nums) {
        Integer c1 = null, c2 = null;
        int n1 = 0, n2 = 0;
        for (int x : nums) {
            if (c1 != null && x == c1) n1++;
            else if (c2 != null && x == c2) n2++;
            else if (n1 == 0) { c1 = x; n1 = 1; }
            else if (n2 == 0) { c2 = x; n2 = 1; }
            else { n1--; n2--; }
        }
        Set<Integer> candidates = new LinkedHashSet<>();
        if (c1 != null) candidates.add(c1);
        if (c2 != null) candidates.add(c2);
        List<Integer> out = new ArrayList<>();
        for (int cand : candidates) {
            int occurrences = 0;
            for (int x : nums) if (x == cand) occurrences++;
            if (occurrences * 3 > nums.length) out.add(cand);
        }
        Collections.sort(out);
        return out;
    }

    public static void main(String[] args) {
        check(majorityElementsN3(new int[]{3, 2, 3}).toString().equals("[3]"), "3,2,3");
        check(majorityElementsN3(new int[]{1, 1, 1, 3, 3, 2, 2, 2}).toString().equals("[1, 2]"), "two values");
        check(majorityElementsN3(new int[]{1, 2, 3}).isEmpty(), "none");
        check(majorityElementsN3(new int[]{}).isEmpty(), "empty");
        check(majorityElementsN3(new int[]{4}).toString().equals("[4]"), "single");
        check(majorityElementsN3(new int[]{1, 1, 2, 2, 3}).toString().equals("[1, 2]"), "2/5 = 0.4 > 1/3");
        check(majorityElementsN3(new int[]{1, 1, 2, 2}).toString().equals("[1, 2]"), "half > 1/3");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
