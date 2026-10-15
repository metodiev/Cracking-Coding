/** Task 293 - Coin change: number of ways (DP). (Tier: Intermediate, CTCI Ch 8) */
import java.util.Arrays;

public class Solution {
    /** Number of unordered combinations of coins (unlimited supply) summing to amount. */
    static long countWays(int[] coins, int amount) {
        long[] ways = new long[amount + 1];
        ways[0] = 1;
        for (int coin : coins) {
            for (int s = coin; s <= amount; s++) ways[s] += ways[s - coin];
        }
        return ways[amount];
    }

    /** Independent oracle: recursive count over coin index with descending multiplicities. */
    static long bruteWays(int[] coins, int amount) {
        return count(coins, 0, amount);
    }

    private static long count(int[] coins, int index, int remaining) {
        if (remaining == 0) return 1;
        if (index == coins.length) return 0;
        long total = 0;
        for (int take = 0; take * coins[index] <= remaining; take++) {
            total += count(coins, index + 1, remaining - take * coins[index]);
        }
        return total;
    }

    private static int seed = 24680;

    private static int[] lcgCoins(int n) {
        int[] values = new int[n];
        int count = 0;
        while (count < n) {
            seed = (seed * 75 + 74) % 65537;
            int value = 1 + seed % 8;
            boolean seen = false;
            for (int i = 0; i < count; i++) if (values[i] == value) seen = true;
            if (!seen) values[count++] = value;
        }
        Arrays.sort(values);
        return values;
    }

    public static void main(String[] args) {
        check(countWays(new int[]{1, 2, 5}, 5) == 4, "1,2,5 amount 5");
        check(countWays(new int[]{2}, 3) == 0, "2 amount 3");
        check(countWays(new int[]{1, 2, 5}, 0) == 1, "amount 0");
        check(countWays(new int[]{1}, 10) == 1, "1 amount 10");
        check(countWays(new int[]{}, 0) == 1, "no coins amount 0");
        check(countWays(new int[]{2, 3, 5}, 9) == 3, "2,3,5 amount 9");
        check(countWays(new int[]{1, 2}, 3) == 2, "order must not matter");

        for (int trial = 0; trial < 150; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 5;
            int[] coins = lcgCoins(count);
            seed = (seed * 75 + 74) % 65537;
            int amount = seed % 60;
            check(countWays(coins, amount) == bruteWays(coins, amount), "random trial " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
