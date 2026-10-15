/** Task 225 - Stock with transaction fee (DP). (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** Unlimited transactions, each sale pays a fixed fee. */
    static int maxProfitFee(int[] prices, int fee) {
        if (fee < 0) throw new IllegalArgumentException("fee must be non-negative");
        if (prices.length == 0) return 0;
        int cash = 0, hold = -prices[0];
        for (int i = 1; i < prices.length; i++) {
            int nextHold = Math.max(hold, cash - prices[i]);
            int nextCash = Math.max(cash, hold + prices[i] - fee);
            hold = nextHold;
            cash = nextCash;
        }
        return cash;
    }

    public static void main(String[] args) {
        check(maxProfitFee(new int[]{1, 3, 2, 8, 4, 9}, 2) == 8, "(8-1-2) + (9-4-2)");
        check(maxProfitFee(new int[]{1, 3, 2, 8, 4, 9}, 0) == 13, "fee 0 matches greedy 223");
        check(maxProfitFee(new int[]{1, 2}, 5) == 0, "fee eats the gain");
        check(maxProfitFee(new int[]{1, 2}, 0) == 1, "free trade");
        check(maxProfitFee(new int[]{1, 4}, 1) == 2, "fee 1");
        check(maxProfitFee(new int[]{}, 2) == 0, "empty");
        check(maxProfitFee(new int[]{5}, 1) == 0, "single");
        check(maxProfitFee(new int[]{5}, 0) == 0, "single with fee 0");

        boolean threw = false;
        try { maxProfitFee(new int[]{1, 2}, -1); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "negative fee rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
