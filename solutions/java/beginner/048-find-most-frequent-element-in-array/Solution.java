/** Task 048 - Find most frequent element in array. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int mostFrequent(int[] items) {
        if (items.length == 0) throw new IllegalArgumentException("empty array");
        java.util.Map<Integer, Integer> counts = new java.util.HashMap<>();
        for (int x : items) counts.merge(x, 1, Integer::sum);
        int best = items[0], bestCount = 0;
        for (int x : items) {
            int c = counts.get(x);
            if (c > bestCount) { best = x; bestCount = c; }
        }
        return best;
    }

    public static void main(String[] args) {
        check(mostFrequent(new int[]{1, 3, 2, 3, 4, 3}) == 3, "basic");
        check(mostFrequent(new int[]{5}) == 5, "single");
        check(mostFrequent(new int[]{1, 2, 1, 2}) == 1, "tie first");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
