/** Task 223 - Stock buy/sell multiple transactions (greedy). (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** Unlimited transactions; sum every positive consecutive gain. */
    static int maxProfitMultiple(int[] prices) {
        int total = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) total += prices[i] - prices[i - 1];
        }
        return total;
    }

    public static void main(String[] args) {
        check(maxProfitMultiple(new int[]{7, 1, 5, 3, 6, 4}) == 7, "(5-1) + (6-3)");
        check(maxProfitMultiple(new int[]{7, 6, 4, 3, 1}) == 0, "monotone decreasing");
        check(maxProfitMultiple(new int[]{1, 2, 3, 4, 5}) == 4, "one long rise");
        check(maxProfitMultiple(new int[]{1, 3, 2, 8}) == 8, "+2 then +6");
        check(maxProfitMultiple(new int[]{}) == 0, "empty");
        check(maxProfitMultiple(new int[]{5}) == 0, "single");
        check(maxProfitMultiple(new int[]{3, 3}) == 0, "flat");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
