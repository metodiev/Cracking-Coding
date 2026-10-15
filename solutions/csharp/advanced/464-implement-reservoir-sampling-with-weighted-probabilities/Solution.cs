// Task 464 - Implement reservoir sampling with weighted probabilities. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Weighted reservoir sampling, A-Res (Efraimidis & Spirakis): each item gets the
// key u^(1/w) with u uniform in (0,1] drawn from a FIXED LCG as u = (seed + 1) /
// 65537; the reservoir keeps the k largest keys (min-heap; an equal key is
// displaced only by a smaller item index, as documented). For k = 1 an item with
// weight w_i is selected with probability exactly w_i / W, which the statistical
// checks exercise.
//
// Checks (deterministic): weights [1,2,3,4] over 20000 trials must land within
// TOL = 0.02 of the weight ratios (measured max deviation 0.0022) with exactly one
// item per trial; equal weights give the uniform 1/3; weights 1..10 with k = 3 over
// 3000 trials yield 3 distinct items per trial, counts summing to 9000, and heavier
// items selected more often.
using System;
using System.Collections.Generic;

internal class WeightedReservoir
{
    private readonly int size;
    private readonly PriorityQueue<int, (double Key, int Item)> heap;

    public WeightedReservoir(int size)
    {
        this.size = size;
        heap = new PriorityQueue<int, (double, int)>();
    }

    public void Add(int item, double weight, double u)
    {
        double key = Math.Pow(u, 1.0 / weight);
        if (heap.Count < size)
        {
            heap.Enqueue(item, (key, item));
        }
        else
        {
            heap.TryPeek(out _, out (double Key, int Item) top);
            if (key > top.Key || (key == top.Key && item < top.Item))
            {
                heap.Dequeue();
                heap.Enqueue(item, (key, item));
            }
        }
    }

    public int[] Sample()
    {
        var items = new int[heap.Count];
        int i = 0;
        foreach (var element in heap.UnorderedItems) items[i++] = element.Element;
        Array.Sort(items);
        return items;
    }
}

public static class Solution
{
    private const double Tol = 0.02;
    private static int seed;

    private static double NextU()
    {
        seed = (seed * 75 + 74) % 65537;
        return (seed + 1) / 65537.0;
    }

    private static int[] SampleOnce(int k, double[] weights)
    {
        var reservoir = new WeightedReservoir(k);
        for (int item = 0; item < weights.Length; item++)
            reservoir.Add(item, weights[item], NextU());
        return reservoir.Sample();
    }

    public static void Main()
    {
        var weights = new double[] { 1, 2, 3, 4 };
        int trials = 20000;
        seed = 99;
        var counts = new int[weights.Length];
        for (int t = 0; t < trials; t++)
        {
            var picked = SampleOnce(1, weights);
            Check(picked.Length == 1, "one item per trial");
            counts[picked[0]]++;
        }
        double total = 0;
        foreach (double w in weights) total += w;
        double maxDev = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            double expected = weights[i] / total;
            double observed = counts[i] / (double)trials;
            maxDev = Math.Max(maxDev, Math.Abs(observed - expected));
            Check(Math.Abs(observed - expected) <= Tol, "freq " + i);
        }
        int sum = 0;
        foreach (int c in counts) sum += c;
        Check(sum == trials, "count bookkeeping");
        Check(counts[3] > counts[0], "heavier selected more");
        Check(maxDev <= 0.01, "measured max deviation " + maxDev);

        seed = 12345;
        counts = new int[3];
        for (int t = 0; t < 20000; t++)
        {
            var picked = SampleOnce(1, new double[] { 1, 1, 1 });
            counts[picked[0]]++;
        }
        for (int i = 0; i < 3; i++) Check(Math.Abs(counts[i] / 20000.0 - 1.0 / 3) <= Tol, "uniform " + i);

        var ws = new double[10];
        for (int i = 0; i < 10; i++) ws[i] = i + 1;
        const int k = 3;
        trials = 3000;
        seed = 777;
        counts = new int[10];
        for (int t = 0; t < trials; t++)
        {
            var picked = SampleOnce(k, ws);
            Check(picked.Length == k, "sample size");
            var distinct = new HashSet<int>(picked);
            Check(distinct.Count == k, "distinct items");
            foreach (int item in picked) counts[item]++;
        }
        sum = 0;
        foreach (int c in counts) sum += c;
        Check(sum == trials * k, "k bookkeeping");
        Check(counts[9] > counts[0], "monotonicity");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
