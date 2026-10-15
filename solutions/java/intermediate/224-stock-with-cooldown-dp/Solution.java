/** Task 224 - Stock with cooldown (DP). (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** Unlimited transactions with a mandatory one-day rest after each sale. */
    static int maxProfitCooldown(int[] prices) {
        if (prices.length == 0) return 0;
        int held = -prices[0];                 // bought on day 0
        int sold = Integer.MIN_VALUE;          // no sale yet
        int rest = 0;                          // idle on day 0
        for (int i = 1; i < prices.length; i++) {
            int nextHeld = Math.max(held, rest - prices[i]);
            int nextSold = held + prices[i];
            int nextRest = Math.max(rest, sold);
            held = nextHeld;
            sold = nextSold;
            rest = nextRest;
        }
        return Math.max(sold, rest);
    }

    public static void main(String[] args) {
        check(maxProfitCooldown(new int[]{1, 2, 3, 0, 2}) == 3, "buy 1/sell 2, rest, buy 0/sell 2");
        check(maxProfitCooldown(new int[]{1}) == 0, "single");
        check(maxProfitCooldown(new int[]{}) == 0, "empty");
        check(maxProfitCooldown(new int[]{2, 1, 4}) == 3, "buy 1 sell 4");
        check(maxProfitCooldown(new int[]{6, 1, 3, 2, 4, 7}) == 6, "hold through, one sale");
        check(maxProfitCooldown(new int[]{1, 2}) == 1, "one up step");
        check(maxProfitCooldown(new int[]{2, 2, 2}) == 0, "flat");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
