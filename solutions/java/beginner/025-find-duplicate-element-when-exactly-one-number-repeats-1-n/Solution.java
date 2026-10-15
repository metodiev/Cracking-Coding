/** Task 025 - Find duplicate element when exactly one number repeats (1..n). (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int findDuplicate(int[] nums) {
        int slow = nums[0], fast = nums[0];
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = nums[0];
        while (slow != fast) { slow = nums[slow]; fast = nums[fast]; }
        return slow;
    }

    public static void main(String[] args) {
        check(findDuplicate(new int[]{1, 3, 4, 2, 2}) == 2, "n=4");
        check(findDuplicate(new int[]{3, 1, 3, 4, 2}) == 3, "value 3");
        check(findDuplicate(new int[]{1, 1}) == 1, "tiny");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
