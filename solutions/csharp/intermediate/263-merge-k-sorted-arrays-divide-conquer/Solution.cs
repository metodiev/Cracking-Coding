// Task 263 - Merge K sorted arrays (divide & conquer). (Tier: Intermediate, CTCI Ch 10)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int[] MergeTwo(int[] a, int[] b)
    {
        var outArr = new int[a.Length + b.Length];
        int i = 0, j = 0, k = 0;
        while (i < a.Length && j < b.Length) outArr[k++] = a[i] <= b[j] ? a[i++] : b[j++];
        while (i < a.Length) outArr[k++] = a[i++];
        while (j < b.Length) outArr[k++] = b[j++];
        return outArr;
    }

    /// <summary>
    /// Pairwise divide-and-conquer merge with a plain 2-way two-pointer merge (NO heap):
    /// log k levels of O(n) work -> O(n log k). Index ranges avoid copying the halves.
    /// k = 0 -> empty; k = 1 returns a fresh copy; inputs are never modified.
    /// </summary>
    public static int[] MergeKDivide(int[][] arrays)
    {
        if (arrays.Length == 0) return new int[0];
        return Divide(arrays, 0, arrays.Length);
    }

    private static int[] Divide(int[][] arrays, int lo, int hi)
    {
        if (hi - lo == 1) return (int[])arrays[lo].Clone();
        int mid = (lo + hi) / 2;
        return MergeTwo(Divide(arrays, lo, mid), Divide(arrays, mid, hi));
    }

    /// <summary>Independent heap-based merge used only for cross-checks (same result expected).</summary>
    public static int[] MergeKHeap(int[][] arrays)
    {
        var pq = new PriorityQueue<(int Val, int Arr, int Pos), (int Val, int Arr, int Pos)>();
        for (int i = 0; i < arrays.Length; i++)
            if (arrays[i].Length > 0) pq.Enqueue((arrays[i][0], i, 0), (arrays[i][0], i, 0));
        var outList = new List<int>();
        while (pq.Count > 0)
        {
            var e = pq.Dequeue();
            outList.Add(e.Val);
            if (e.Pos + 1 < arrays[e.Arr].Length)
            {
                var next = (arrays[e.Arr][e.Pos + 1], e.Arr, e.Pos + 1);
                pq.Enqueue(next, next);
            }
        }
        return outList.ToArray();
    }

    /// <summary>Flat sort reference used only for cross-checks.</summary>
    public static int[] MergeOracle(int[][] arrays)
    {
        var values = new List<int>();
        foreach (var arr in arrays) values.AddRange(arr);
        values.Sort();
        return values.ToArray();
    }

    private static void CheckArray(int[] got, int[] expected, string label)
    {
        bool ok = got.Length == expected.Length;
        for (int i = 0; ok && i < expected.Length; i++) ok = got[i] == expected[i];
        Check(ok, label + " got=[" + string.Join(", ", got) + "]");
    }

    public static void Main()
    {
        var classic = new[] { new[] { 1, 4, 5 }, new[] { 1, 3, 4 }, new[] { 2, 6 } };
        CheckArray(MergeKDivide(classic), new[] { 1, 1, 2, 3, 4, 4, 5, 6 }, "classic");
        CheckArray(MergeKDivide(new int[0][]), new int[0], "k = 0");
        CheckArray(MergeKDivide(new[] { new[] { 3, 4 } }), new[] { 3, 4 }, "k = 1");
        CheckArray(MergeKDivide(new[] { new int[0] }), new int[0], "single empty");
        CheckArray(MergeKDivide(new[] { new[] { 1 }, new[] { 2, 3, 4, 5 }, new[] { 6 } }),
            new[] { 1, 2, 3, 4, 5, 6 }, "unequal lengths");
        CheckArray(MergeKDivide(new[] { new[] { 2, 2 }, new[] { 2, 2 }, new[] { 2 } }),
            new[] { 2, 2, 2, 2, 2 }, "duplicates");
        CheckArray(MergeKDivide(new[] { new int[0], new[] { 1 }, new int[0], new[] { 2, 3 } }),
            new[] { 1, 2, 3 }, "empty arrays inside");
        CheckArray(MergeKDivide(new[] { new[] { -3, -1 }, new[] { -2, 0 } }),
            new[] { -3, -2, -1, 0 }, "negatives");

        var original = new[] { 3, 4 };
        var copy = MergeKDivide(new[] { original });
        bool sameValues = copy.Length == original.Length && copy[0] == 3 && copy[1] == 4;
        Check(sameValues && !ReferenceEquals(copy, original), "k = 1 returns a fresh copy");

        var fixedCases = new[]
        {
            classic, new int[0][], new[] { new[] { 3, 4 } }, new[] { new int[0] },
            new[] { new[] { 1 }, new[] { 2, 3, 4, 5 }, new[] { 6 } },
            new[] { new[] { 2, 2 }, new[] { 2, 2 }, new[] { 2 } },
            new[] { new[] { -3, -1 }, new[] { -2, 0 } },
        };
        foreach (var arrays in fixedCases)          // divide & conquer == heap == flat sort
        {
            CheckArray(MergeKDivide(arrays), MergeKHeap(arrays), "divide == heap");
            CheckArray(MergeKDivide(arrays), MergeOracle(arrays), "divide == oracle");
        }

        var rng = new Random(263);
        for (int test = 0; test < 300; test++)
        {
            int k = rng.Next(7);
            var arrays = new int[k][];
            for (int i = 0; i < k; i++)
            {
                var arr = new int[rng.Next(7)];
                for (int j = 0; j < arr.Length; j++) arr[j] = rng.Next(13) - 6;
                Array.Sort(arr);
                arrays[i] = arr;
            }
            CheckArray(MergeKDivide(arrays), MergeKHeap(arrays), "random divide == heap");
            CheckArray(MergeKDivide(arrays), MergeOracle(arrays), "random divide == oracle");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
