/** Task 220 - Implement bitmask DP for TSP small N. (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    static long tsp(int[][] dist) {
        int n = dist.length;
        if (n <= 1) return 0;
        long INF = Long.MAX_VALUE / 4;
        int full = (1 << n) - 1;
        long[][] dp = new long[1 << n][n];
        for (long[] row : dp) java.util.Arrays.fill(row, INF);
        dp[1][0] = 0;
        for (int mask = 0; mask <= full; mask++)
            for (int last = 0; last < n; last++) {
                if (dp[mask][last] == INF) continue;
                for (int nxt = 0; nxt < n; nxt++) {
                    if ((mask & (1 << nxt)) != 0) continue;
                    int newMask = mask | (1 << nxt);
                    long candidate = dp[mask][last] + dist[last][nxt];
                    if (candidate < dp[newMask][nxt]) dp[newMask][nxt] = candidate;
                }
            }
        long best = INF;
        for (int last = 0; last < n; last++)
            if (dp[full][last] != INF) best = Math.min(best, dp[full][last] + dist[last][0]);
        return best;
    }

    public static void main(String[] args) {
        int[][] square = {
            {0, 1, 10, 10},
            {1, 0, 1, 10},
            {10, 1, 0, 1},
            {1, 10, 1, 0},
        };
        check(tsp(square) == 4, "square loop, all cycle edges 1");

        int[][] triangle = {{0, 1, 3}, {1, 0, 2}, {3, 2, 0}};
        check(tsp(triangle) == 6, "triangle");

        check(tsp(new int[][]{{0}}) == 0, "single city");
        check(tsp(new int[][]{{0, 7}, {7, 0}}) == 14, "two cities");

        // brute-force cross-check on a random 7-city instance
        java.util.Random rng = new java.util.Random(6);
        int n = 7;
        int[][] d = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) d[i][j] = d[j][i] = 1 + rng.nextInt(19);
        int[] perm = {1, 2, 3, 4, 5, 6};
        long brute = Long.MAX_VALUE;
        do {
            long cost = 0;
            int prev = 0;
            for (int city : perm) {
                cost += d[prev][city];
                prev = city;
            }
            cost += d[prev][0];
            brute = Math.min(brute, cost);
        } while (nextPermutation(perm));
        check(tsp(d) == brute, "matches brute force: " + tsp(d) + " vs " + brute);
        System.out.println("PASS");
    }

    private static boolean nextPermutation(int[] a) {
        int i = a.length - 2;
        while (i >= 0 && a[i] >= a[i + 1]) i--;
        if (i < 0) return false;
        int j = a.length - 1;
        while (a[j] <= a[i]) j--;
        int t = a[i]; a[i] = a[j]; a[j] = t;
        for (int lo = i + 1, hi = a.length - 1; lo < hi; lo++, hi--) { t = a[lo]; a[lo] = a[hi]; a[hi] = t; }
        return true;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
