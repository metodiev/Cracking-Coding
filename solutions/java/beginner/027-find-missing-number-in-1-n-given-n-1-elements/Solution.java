/** Task 027 - Find missing number in 1..n given n-1 elements. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int missingNumber(int[] nums, int n) {
        long expected = (long) n * (n + 1) / 2;
        long actual = 0;
        for (int x : nums) actual += x;
        return (int) (expected - actual);
    }

    public static void main(String[] args) {
        check(missingNumber(new int[]{1, 2, 4, 5}, 5) == 3, "middle");
        check(missingNumber(new int[]{2}, 2) == 1, "first");
        check(missingNumber(new int[]{1}, 2) == 2, "last");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
