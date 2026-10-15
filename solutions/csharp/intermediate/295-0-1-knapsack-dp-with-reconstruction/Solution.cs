// Task 295 - 0/1 knapsack (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Returns (best value, chosen indices): DP over (item prefix, capacity) + backtrack.</summary>
    public static (int Value, int[] Chosen) Knapsack(int[] weights, int[] values, int capacity)
    {
        int n = weights.Length;
        var dp = new int[n + 1, capacity + 1];
        for (int i = 1; i <= n; i++)
        {
            int weight = weights[i - 1];
            int value = values[i - 1];
            for (int c = 0; c <= capacity; c++)
            {
                int best = dp[i - 1, c];
                if (c >= weight && dp[i - 1, c - weight] + value > best) best = dp[i - 1, c - weight] + value;
                dp[i, c] = best;
            }
        }
        var reversed = new List<int>();
        int rest = capacity;
        for (int i = n; i >= 1; i--)
        {
            if (dp[i, rest] != dp[i - 1, rest])
            {
                reversed.Add(i - 1);
                rest -= weights[i - 1];
            }
        }
        reversed.Reverse();
        return (dp[n, capacity], reversed.ToArray());
    }

    public static int BruteKnapsack(int[] weights, int[] values, int capacity)
    {
        int n = weights.Length;
        int best = 0;
        for (int mask = 0; mask < (1 << n); mask++)
        {
            int weight = 0;
            int value = 0;
            for (int i = 0; i < n; i++)
            {
                if (((mask >> i) & 1) != 0)
                {
                    weight += weights[i];
                    value += values[i];
                }
            }
            if (weight <= capacity && value > best) best = value;
        }
        return best;
    }

    private static int seed = 555;

    private static (int[] Weights, int[] Values) LcgInstance(int n)
    {
        var weights = new int[n];
        var values = new int[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            weights[i] = 1 + seed % 12;
            seed = (seed * 75 + 74) % 65537;
            values[i] = 1 + seed % 12;
        }
        return (weights, values);
    }

    public static void Main()
    {
        var weights = new[] { 1, 3, 4, 5 };
        var values = new[] { 1, 4, 5, 7 };
        var (value, chosen) = Knapsack(weights, values, 7);
        Check(value == 9 && chosen.Length == 2, "hand-solved value");
        Check(weights[chosen[0]] + weights[chosen[1]] == 7 && values[chosen[0]] + values[chosen[1]] == 9,
            "hand-solved set (3 and 4)");

        var empty = Knapsack(weights, values, 0);
        Check(empty.Value == 0 && empty.Chosen.Length == 0, "capacity 0");
        var heavy = Knapsack(new[] { 2, 9 }, new[] { 3, 100 }, 5);
        Check(heavy.Value == 3 && heavy.Chosen.Length == 1 && heavy.Chosen[0] == 0, "heavier item skipped");
        var exact = Knapsack(new[] { 4 }, new[] { 9 }, 4);
        Check(exact.Value == 9 && exact.Chosen.Length == 1 && exact.Chosen[0] == 0, "single item fits");

        var all = Knapsack(weights, values, 100);
        Check(all.Value == 17 && all.Chosen.Length == 4, "everything fits");

        for (int trial = 0; trial < 120; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 14;
            var (w, v) = LcgInstance(n);
            seed = (seed * 75 + 74) % 65537;
            int capacity = seed % 40;
            var (answer, picked) = Knapsack(w, v, capacity);
            Check(answer == BruteKnapsack(w, v, capacity), "random trial " + trial);
            int weight = 0;
            int total = 0;
            foreach (int index in picked)
            {
                weight += w[index];
                total += v[index];
            }
            Check(weight <= capacity && total == answer, "random reconstruction " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
