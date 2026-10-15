/** Task 295 - 0/1 knapsack (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** Returns {best value, chosen indices...}: dp over (item prefix, capacity) + backtrack. */
    static int[] knapsack(int[] weights, int[] values, int capacity) {
        int n = weights.length;
        int[][] dp = new int[n + 1][capacity + 1];
        for (int i = 1; i <= n; i++) {
            int weight = weights[i - 1];
            int value = values[i - 1];
            for (int c = 0; c <= capacity; c++) {
                int best = dp[i - 1][c];
                if (c >= weight && dp[i - 1][c - weight] + value > best) best = dp[i - 1][c - weight] + value;
                dp[i][c] = best;
            }
        }
        int[] reversed = new int[n];
        int count = 0;
        int c = capacity;
        for (int i = n; i >= 1; i--) {
            if (dp[i][c] != dp[i - 1][c]) {
                reversed[count++] = i - 1;
                c -= weights[i - 1];
            }
        }
        int[] result = new int[count + 1];
        result[0] = dp[n][capacity];
        for (int i = 0; i < count; i++) result[i + 1] = reversed[count - 1 - i];
        return result;
    }

    static int bruteKnapsack(int[] weights, int[] values, int capacity) {
        int n = weights.length;
        int best = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            int weight = 0;
            int value = 0;
            for (int i = 0; i < n; i++) {
                if (((mask >> i) & 1) != 0) {
                    weight += weights[i];
                    value += values[i];
                }
            }
            if (weight <= capacity && value > best) best = value;
        }
        return best;
    }

    private static int seed = 555;

    private static int[][] lcgInstance(int n) {
        int[] weights = new int[n];
        int[] values = new int[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            weights[i] = 1 + seed % 12;
            seed = (seed * 75 + 74) % 65537;
            values[i] = 1 + seed % 12;
        }
        return new int[][]{weights, values};
    }

    public static void main(String[] args) {
        int[] weights = new int[]{1, 3, 4, 5};
        int[] values = new int[]{1, 4, 5, 7};
        int[] result = knapsack(weights, values, 7);
        check(result[0] == 9 && result.length == 3, "hand-solved value");
        int chosenWeight = weights[result[1]] + weights[result[2]];
        int chosenValue = values[result[1]] + values[result[2]];
        check(chosenWeight == 7 && chosenValue == 9, "hand-solved set (3 and 4)");

        int[] empty = knapsack(weights, values, 0);
        check(empty.length == 1 && empty[0] == 0, "capacity 0");
        int[] heavy = knapsack(new int[]{2, 9}, new int[]{3, 100}, 5);
        check(heavy.length == 2 && heavy[0] == 3 && heavy[1] == 0, "heavier item skipped");
        int[] exact = knapsack(new int[]{4}, new int[]{9}, 4);
        check(exact.length == 2 && exact[0] == 9 && exact[1] == 0, "single item fits");

        int[] all = knapsack(weights, values, 100);
        check(all[0] == 17 && all.length == 5, "everything fits");

        for (int trial = 0; trial < 120; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 14;
            int[][] instance = lcgInstance(n);
            int[] w = instance[0];
            int[] v = instance[1];
            seed = (seed * 75 + 74) % 65537;
            int capacity = seed % 40;
            int[] answer = knapsack(w, v, capacity);
            check(answer[0] == bruteKnapsack(w, v, capacity), "random trial " + trial);
            int weight = 0;
            int value = 0;
            for (int i = 1; i < answer.length; i++) {
                weight += w[answer[i]];
                value += v[answer[i]];
            }
            check(weight <= capacity && value == answer[0], "random reconstruction " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
