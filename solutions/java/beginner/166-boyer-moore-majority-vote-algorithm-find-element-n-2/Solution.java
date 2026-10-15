/** Task 166 - Boyer-Moore majority vote algorithm (element > n/2). (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    /** Returns the element occurring more than n/2 times, or null. */
    static Integer majorityElement(int[] nums) {
        Integer candidate = null;
        int count = 0;
        for (int x : nums) {
            if (count == 0) {
                candidate = x;
                count = 1;
            } else if (x == candidate) {
                count++;
            } else {
                count--;
            }
        }
        if (candidate == null) return null;
        int occurrences = 0;
        for (int x : nums) if (x == candidate) occurrences++;
        return occurrences > nums.length / 2 ? candidate : null;
    }

    public static void main(String[] args) {
        check(majorityElement(new int[]{3, 2, 3}) == 3, "3,2,3");
        check(majorityElement(new int[]{2, 2, 1, 1, 1, 2, 2}) == 2, "classic");
        check(majorityElement(new int[]{1}) == 1, "single");
        check(majorityElement(new int[]{}) == null, "empty");
        check(majorityElement(new int[]{1, 2, 3}) == null, "no majority");
        check(majorityElement(new int[]{1, 1, 2, 2}) == null, "exactly half is not majority");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
