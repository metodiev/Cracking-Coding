// Task 339 - Range minimum query with offline queries (Mo's algorithm simple). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

public class LazyMinMultiset
{
    // Minimum of the swept window: a min-heap with lazily deleted values; each remove marks
    // one occurrence, and the mark is consumed when that value reaches the heap top.
    private readonly PriorityQueue<int, int> _heap = new PriorityQueue<int, int>();
    private readonly Dictionary<int, int> _pending = new Dictionary<int, int>();
    private int _size = 0;

    public void Add(int value)
    {
        _heap.Enqueue(value, value);
        _size++;
    }

    public void Remove(int value)
    {
        _pending[value] = Pending(value) + 1;
        _size--;
    }

    public int Minimum()
    {
        while (_heap.Count > 0 && Pending(_heap.Peek()) > 0)
        {
            int value = _heap.Dequeue();
            _pending[value] = _pending[value] - 1;
        }
        if (_size <= 0) throw new Exception("minimum of an empty multiset");
        return _heap.Peek();
    }

    private int Pending(int value)
    {
        return _pending.TryGetValue(value, out int count) ? count : 0;
    }
}

public static class Solution
{
    /// <summary>
    /// Simplified Mo's algorithm: block-sort the queries by (l / B, r) with B = n/sqrt(q), then
    /// sweep the window boundaries. The sweep performs O((n + q) * sqrt(n)) add/remove moves
    /// (the classic Mo bound), each O(log n) amortised on the lazy min-heap.
    /// </summary>
    public static int[] MoRangeMin(int[] values, int[][] queries)
    {
        int n = values.Length;
        int q = queries.Length;
        if (q == 0) return Array.Empty<int>();
        int root = (int)Math.Sqrt(q);
        int block = root > 0 ? Math.Max(1, n / root) : n;

        var order = new int[q];
        for (int i = 0; i < q; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int blockA = queries[a][0] / block;
            int blockB = queries[b][0] / block;
            if (blockA != blockB) return blockA - blockB;
            int rightA = blockA % 2 == 0 ? queries[a][1] : -queries[a][1];
            int rightB = blockB % 2 == 0 ? queries[b][1] : -queries[b][1];
            return rightA - rightB;
        });

        var window = new LazyMinMultiset();
        var answers = new int[q];
        int currentLeft = 0, currentRight = -1;
        foreach (int index in order)
        {
            int left = queries[index][0];
            int right = queries[index][1];
            while (currentLeft > left) window.Add(values[--currentLeft]);
            while (currentRight < right) window.Add(values[++currentRight]);
            while (currentLeft < left) window.Remove(values[currentLeft++]);
            while (currentRight > right) window.Remove(values[currentRight--]);
            answers[index] = window.Minimum();
        }
        return answers;
    }

    public static int[] NaiveRangeMin(int[] values, int[][] queries)
    {
        var answers = new int[queries.Length];
        for (int i = 0; i < queries.Length; i++)
        {
            int best = values[queries[i][0]];
            for (int j = queries[i][0] + 1; j <= queries[i][1]; j++) best = Math.Min(best, values[j]);
            answers[i] = best;
        }
        return answers;
    }

    private static int seed = 5150;

    private static string Join(int[] values)
    {
        return string.Join(",", values);
    }

    public static void Main()
    {
        // Hand-checked minima on a small fixed array.
        var values = new[] { 1, 2, 1, 3, 2, 4 };
        var queries = new[]
        {
            new[] { 0, 2 }, new[] { 1, 4 }, new[] { 0, 5 }, new[] { 3, 3 },
            new[] { 2, 5 }, new[] { 1, 1 }, new[] { 5, 5 },
        };
        Check(Join(MoRangeMin(values, queries)) == "1,1,1,3,1,2,4", "hand answers");
        Check(Join(NaiveRangeMin(values, queries)) == "1,1,1,3,1,2,4", "hand naive");
        Check(Join(MoRangeMin(new[] { 7 }, new[] { new[] { 0, 0 } })) == "7", "single element");
        Check(MoRangeMin(values, Array.Empty<int[]>()).Length == 0, "no queries");

        // All-equal and strictly increasing arrays, including duplicate-heavy windows.
        var flat = new[] { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
        Check(Join(MoRangeMin(flat, new[] { new[] { 0, 9 }, new[] { 2, 3 }, new[] { 4, 4 } })) == "5,5,5",
            "flat");
        var rising = new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        Check(Join(MoRangeMin(rising, new[] { new[] { 0, 9 }, new[] { 0, 0 }, new[] { 3, 7 } })) == "0,0,3",
            "rising");
        var twos = new[] { 2, 2, 3, 2, 3, 3 };
        Check(Join(MoRangeMin(twos, new[] { new[] { 0, 5 }, new[] { 1, 4 }, new[] { 2, 5 }, new[] { 3, 5 } })) == "2,2,2,2",
            "duplicate-heavy");

        // LCG cross-check with a spread of sizes and query counts.
        for (int trial = 0; trial < 12; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 120;
            seed = (seed * 75 + 74) % 65537;
            int q = 1 + seed % 200;
            seed = (seed * 75 + 74) % 65537;
            int alphabet = 1 + seed % 12;
            var randomValues = new int[n];
            for (int i = 0; i < n; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                randomValues[i] = seed % alphabet;
            }
            var randomQueries = new int[q][];
            for (int i = 0; i < q; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                int left = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int right = left + seed % (n - left);
                randomQueries[i] = new[] { left, right };
            }
            Check(Join(MoRangeMin(randomValues, randomQueries))
                == Join(NaiveRangeMin(randomValues, randomQueries)), "random " + trial);
        }

        // Single-position queries, one full query, and an adversarially reversed query order.
        seed = 424242;
        var singlesValues = new int[50];
        for (int i = 0; i < 50; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            singlesValues[i] = seed % 7;
        }
        var singles = new int[50][];
        for (int i = 0; i < 50; i++) singles[i] = new[] { i, i };
        Check(Join(MoRangeMin(singlesValues, singles)) == Join(singlesValues), "single positions");
        var full = new[] { new[] { 0, 49 } };
        Check(Join(MoRangeMin(singlesValues, full)) == Join(NaiveRangeMin(singlesValues, full)),
            "full range");
        var reversed = new int[50][];
        for (int i = 0; i < 50; i++) reversed[i] = new[] { 49 - i, 49 - i };
        Check(Join(MoRangeMin(singlesValues, reversed)) == Join(NaiveRangeMin(singlesValues, reversed)),
            "reversed order");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
