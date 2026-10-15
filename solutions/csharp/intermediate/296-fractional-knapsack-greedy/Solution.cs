// Task 296 - Fractional knapsack (greedy). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Greedy by ratio. Returns (total value, picks as (index, fraction) pairs).</summary>
    public static (double Total, List<(int Index, double Fraction)> Picks) FractionalKnapsack(
        int[] weights, int[] values, int capacity)
    {
        int n = weights.Length;
        var order = new List<int>();
        for (int i = 0; i < n; i++) order.Add(i);
        order.Sort((a, b) =>
        {
            double left = (double)values[a] / weights[a];
            double right = (double)values[b] / weights[b];
            if (left != right) return right.CompareTo(left);
            return a.CompareTo(b);
        });
        double remaining = capacity;
        double total = 0;
        var picks = new List<(int, double)>();
        foreach (int index in order)
        {
            if (remaining <= 0) break;
            double fraction = Math.Min(1.0, remaining / weights[index]);
            total += fraction * values[index];
            remaining -= fraction * weights[index];
            picks.Add((index, fraction));
        }
        return (total, picks);
    }

    /// <summary>
    /// Oracle: any optimum takes whole items plus at most one fraction; enumerate both.
    /// The fraction is capped at 1 because the remaining item exists only once -- omitting
    /// the cap turns the oracle into the unbounded (fluid-supply) variant by mistake.
    /// </summary>
    public static double BruteFractional(int[] weights, int[] values, int capacity)
    {
        int n = weights.Length;
        double best = 0;
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
            if (weight > capacity) continue;
            best = Math.Max(best, value);
            for (int j = 0; j < n; j++)
            {
                if (((mask >> j) & 1) == 0)
                {
                    double fraction = Math.Min(1.0, (double)(capacity - weight) / weights[j]);
                    best = Math.Max(best, value + fraction * values[j]);
                }
            }
        }
        return best;
    }

    private static int seed = 424242;

    private static (int[] Weights, int[] Values) LcgInstance(int n)
    {
        var weights = new int[n];
        var values = new int[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            weights[i] = 1 + seed % 9;
            seed = (seed * 75 + 74) % 65537;
            values[i] = 1 + seed % 9;
        }
        return (weights, values);
    }

    public static void Main()
    {
        var weights = new[] { 10, 20, 30 };
        var values = new[] { 60, 100, 120 };
        var (total, picks) = FractionalKnapsack(weights, values, 50);
        Check(Math.Abs(total - 240.0) < 1e-9, "classic example value");
        Check(picks.Count == 3 && picks[0].Index == 0 && picks[1].Index == 1 && picks[2].Index == 2, "picked indexes");
        Check(Math.Abs(picks[0].Fraction - 1.0) < 1e-9 && Math.Abs(picks[1].Fraction - 1.0) < 1e-9, "whole items");
        Check(Math.Abs(picks[2].Fraction - 2.0 / 3.0) < 1e-9, "two thirds of item 2");

        (total, picks) = FractionalKnapsack(weights, values, 60);
        Check(Math.Abs(total - 280.0) < 1e-9 && picks.Count == 3, "capacity covers everything");

        (total, picks) = FractionalKnapsack(weights, values, 0);
        Check(total == 0.0 && picks.Count == 0, "capacity 0");

        (total, picks) = FractionalKnapsack(new[] { 5 }, new[] { 3 }, 2);
        Check(Math.Abs(total - 1.2) < 1e-9 && Math.Abs(picks[0].Fraction - 0.4) < 1e-9, "single fractional item");

        for (int trial = 0; trial < 120; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 8;
            var (w, v) = LcgInstance(n);
            seed = (seed * 75 + 74) % 65537;
            int capacity = seed % 50;
            var (answer, chosen) = FractionalKnapsack(w, v, capacity);
            Check(Math.Abs(answer - BruteFractional(w, v, capacity)) < 1e-9, "random trial " + trial);
            double weight = 0;
            double value = 0;
            foreach (var (index, fraction) in chosen)
            {
                weight += fraction * w[index];
                value += fraction * v[index];
            }
            Check(weight <= capacity + 1e-9 && Math.Abs(value - answer) < 1e-9, "random picks " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
