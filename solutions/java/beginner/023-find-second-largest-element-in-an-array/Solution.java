/** Task 023 - Find second largest element in an array. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int secondLargest(int[] items) {
        Integer best = null, second = null;
        for (int x : items) {
            if (best == null || x > best) { second = best; best = x; }
            else if (x != best && (second == null || x > second)) second = x;
        }
        if (second == null) throw new IllegalArgumentException("no second largest");
        return second;
    }

    public static void main(String[] args) {
        check(secondLargest(new int[]{3, 7, 2, 9, 4}) == 7, "basic");
        check(secondLargest(new int[]{5, 5, 4}) == 4, "duplicate max");
        check(secondLargest(new int[]{-1, -2, -3}) == -2, "negatives");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
