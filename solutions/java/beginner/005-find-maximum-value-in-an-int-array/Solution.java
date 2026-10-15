/** Task 005 - Find maximum value in an int array. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int maxValue(int[] items) {
        if (items.length == 0) throw new IllegalArgumentException("empty array");
        int best = items[0];
        for (int x : items) if (x > best) best = x;
        return best;
    }

    public static void main(String[] args) {
        check(maxValue(new int[]{3, 7, 2, 9, 4}) == 9, "basic");
        check(maxValue(new int[]{-5, -1, -3}) == -1, "negatives");
        check(maxValue(new int[]{42}) == 42, "single");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
