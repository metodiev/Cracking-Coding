/** Task 294 - Coin change: minimum coins (DP). (Tier: Intermediate, CTCI Ch 8) */
import java.util.Arrays;

public class Solution {
    /** Fewest coins summing to amount, or -1 when impossible (documented sentinel). */
    static int minCoins(int[] coins, int amount) {
        int infinity = Integer.MAX_VALUE / 4;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, infinity);
        dp[0] = 0;
        for (int s = 1; s <= amount; s++) {
            for (int coin : coins) {
                if (coin <= s && dp[s - coin] + 1 < dp[s]) dp[s] = dp[s - coin] + 1;
            }
        }
        return dp[amount] == infinity ? -1 : dp[amount];
    }

    /** Independent oracle: BFS over reachable amounts gives fewest-coin distances. */
    static int bfsMinCoins(int[] coins, int amount) {
        int[] dist = new int[amount + 1];
        Arrays.fill(dist, -1);
        dist[0] = 0;
        int[] queue = new int[amount + 1];
        int head = 0;
        int tail = 1;
        while (head < tail) {
            int u = queue[head++];
            for (int coin : coins) {
                int v = u + coin;
                if (v <= amount && dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    queue[tail++] = v;
                }
            }
        }
        return dist[amount];
    }

    private static int seed = 13579;

    private static int[] lcgCoins(int n) {
        int[] values = new int[n];
        int count = 0;
        while (count < n) {
            seed = (seed * 75 + 74) % 65537;
            int value = 1 + seed % 7;
            boolean seen = false;
            for (int i = 0; i < count; i++) if (values[i] == value) seen = true;
            if (!seen) values[count++] = value;
        }
        Arrays.sort(values);
        return values;
    }

    public static void main(String[] args) {
        check(minCoins(new int[]{1, 2, 5}, 11) == 3, "1,2,5 amount 11");
        check(minCoins(new int[]{2}, 3) == -1, "impossible sentinel");
        check(minCoins(new int[]{1, 2, 5}, 0) == 0, "amount 0");
        check(minCoins(new int[]{1, 3, 4}, 6) == 2, "greedy counterexample");
        check(minCoins(new int[]{1, 3, 4}, 9) == 3, "1,3,4 amount 9");
        check(minCoins(new int[]{5, 7}, 11) == -1, "5,7 amount 11");
        check(minCoins(new int[]{}, 0) == 0, "no coins amount 0");

        for (int trial = 0; trial < 150; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 4;
            int[] coins = lcgCoins(count);
            seed = (seed * 75 + 74) % 65537;
            int amount = seed % 80;
            check(minCoins(coins, amount) == bfsMinCoins(coins, amount), "random trial " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
