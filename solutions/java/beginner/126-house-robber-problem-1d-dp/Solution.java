/** Task 126 - House robber problem (1D DP). (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static long rob(int[] houses) {
        long take = 0, skip = 0;
        for (int value : houses) {
            long newTake = skip + value;
            skip = Math.max(skip, take);
            take = newTake;
        }
        return Math.max(take, skip);
    }

    public static void main(String[] args) {
        check(rob(new int[]{2, 7, 9, 3, 1}) == 12, "classic");
        check(rob(new int[]{}) == 0, "empty");
        check(rob(new int[]{5}) == 5, "single");
        check(rob(new int[]{2, 1, 1, 2}) == 4, "both ends");
        check(rob(new int[]{5, 1, 1, 5}) == 10, "non-adjacent ends");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
