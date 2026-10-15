/** Task 206 - Pair with sum closest to target in sorted array. (Tier: Intermediate, CTCI Ch 16) */
public class Solution {
    /** Returns {x, y, sum}; on equal distance the first pair found wins. */
    static int[] closestPair(int[] a, int target) {
        if (a.length < 2) throw new IllegalArgumentException("need at least two values");
        int lo = 0, hi = a.length - 1;
        int bestX = a[lo], bestY = a[hi];
        int bestDiff = Math.abs(bestX + bestY - target);
        while (lo < hi) {
            int total = a[lo] + a[hi];
            int diff = Math.abs(total - target);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestX = a[lo];
                bestY = a[hi];
            }
            if (total < target) lo++;
            else if (total > target) hi--;
            else break;
        }
        return new int[]{bestX, bestY, bestX + bestY};
    }

    public static void main(String[] args) {
        int[] a = {1, 3, 4, 5, 7};
        check(render(closestPair(a, 10)).equals("(3, 7) sum 10"), "exact");
        check(render(closestPair(a, 8)).equals("(1, 7) sum 8"), "exact 2");
        check(render(closestPair(a, 12)).equals("(5, 7) sum 12"), "exact 3");
        check(render(closestPair(a, 0)).equals("(1, 3) sum 4"), "closest below");
        check(render(closestPair(new int[]{2, 4}, 5)).equals("(2, 4) sum 6"), "two values");

        boolean threw = false;
        try { closestPair(new int[]{1}, 5); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "needs two values");
        System.out.println("PASS");
    }

    private static String render(int[] r) {
        return "(" + r[0] + ", " + r[1] + ") sum " + r[2];
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
