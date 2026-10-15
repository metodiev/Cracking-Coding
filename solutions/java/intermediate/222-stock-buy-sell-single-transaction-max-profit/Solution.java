/** Task 222 - Stock buy/sell single transaction max profit. (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** One buy and one sell (buy before sell); 0 when no profit is possible. */
    static int maxProfitSingle(int[] prices) {
        int best = 0, cheapest = Integer.MAX_VALUE;
        for (int p : prices) {
            cheapest = Math.min(cheapest, p);
            best = Math.max(best, p - cheapest);
        }
        return best;
    }

    public static void main(String[] args) {
        check(maxProfitSingle(new int[]{7, 1, 5, 3, 6, 4}) == 5, "buy 1 sell 6");
        check(maxProfitSingle(new int[]{7, 6, 4, 3, 1}) == 0, "only falls");
        check(maxProfitSingle(new int[]{1}) == 0, "single day");
        check(maxProfitSingle(new int[]{}) == 0, "no days");
        check(maxProfitSingle(new int[]{1, 2}) == 1, "one up step");
        check(maxProfitSingle(new int[]{2, 1}) == 0, "one down step");
        check(maxProfitSingle(new int[]{3, 3}) == 0, "flat");
        check(maxProfitSingle(new int[]{5, 4, 6, 2, 7}) == 5, "buy 2 sell 7");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
