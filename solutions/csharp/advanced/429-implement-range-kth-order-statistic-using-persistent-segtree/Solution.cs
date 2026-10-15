// Task 429 - Implement range kth order statistic using persistent segtree. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal class PersistentKthSmallest
{
    private readonly int[] sortedValues;
    private readonly int domain;
    private readonly List<int> left = new List<int>();
    private readonly List<int> right = new List<int>();
    private readonly List<int> count = new List<int>();
    private readonly List<int> roots = new List<int>();

    public PersistentKthSmallest(int[] values)
    {
        var sorted = (int[])values.Clone();
        Array.Sort(sorted);
        var distinct = new List<int>();
        foreach (int value in sorted)
        {
            if (distinct.Count == 0 || distinct[distinct.Count - 1] != value) distinct.Add(value);
        }
        sortedValues = distinct.ToArray();
        domain = distinct.Count;
        var rankOf = new Dictionary<int, int>();
        for (int i = 0; i < domain; i++) rankOf[sortedValues[i]] = i;
        left.Add(0);
        right.Add(0);
        count.Add(0);
        int root = 0;
        roots.Add(0);
        foreach (int value in values)
        {
            root = Insert(root, 0, domain - 1, rankOf[value]);
            roots.Add(root);
        }
    }

    private int MakeNode(int l, int r, int c)
    {
        left.Add(l);
        right.Add(r);
        count.Add(c);
        return count.Count - 1;
    }

    private int Insert(int node, int lo, int hi, int position)
    {
        if (lo == hi) return MakeNode(0, 0, count[node] + 1);
        int mid = (lo + hi) / 2;
        if (position <= mid)
        {
            int l = Insert(left[node], lo, mid, position);
            int r = right[node];
            return MakeNode(l, r, count[l] + count[r]);
        }
        int leftChild = left[node];
        int rightChild = Insert(right[node], mid + 1, hi, position);
        return MakeNode(leftChild, rightChild, count[leftChild] + count[rightChild]);
    }

    public int Kth(int from, int to, int k)
    {
        int low = roots[from];
        int high = roots[to + 1];
        int lo = 0, hi = domain - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            int leftCount = count[left[high]] - count[left[low]];
            if (k <= leftCount)
            {
                low = left[low];
                high = left[high];
                hi = mid;
            }
            else
            {
                k -= leftCount;
                low = right[low];
                high = right[high];
                lo = mid + 1;
            }
        }
        return sortedValues[lo];
    }
}

public static class Solution
{
    private static int seed;

    private static int[] LcgArray(int n)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % 13 - 6;
        }
        return values;
    }

    public static void Main()
    {
        var tree = new PersistentKthSmallest(new[] { 5, 1, 4, 2, 3 });
        Check(tree.Kth(0, 4, 1) == 1, "whole 1st");
        Check(tree.Kth(0, 4, 2) == 2, "whole 2nd");
        Check(tree.Kth(0, 4, 3) == 3, "whole 3rd");
        Check(tree.Kth(0, 4, 4) == 4, "whole 4th");
        Check(tree.Kth(0, 4, 5) == 5, "whole 5th");
        Check(tree.Kth(1, 3, 1) == 1, "middle 1st");
        Check(tree.Kth(1, 3, 2) == 2, "middle 2nd");
        Check(tree.Kth(1, 3, 3) == 4, "middle 3rd");
        Check(tree.Kth(2, 2, 1) == 4, "cell");
        Check(tree.Kth(4, 4, 1) == 3, "last cell");
        Check(tree.Kth(0, 0, 1) == 5, "first cell");

        var dupes = new PersistentKthSmallest(new[] { 2, 2, 2 });
        Check(dupes.Kth(0, 2, 1) == 2, "dupes 1st");
        Check(dupes.Kth(0, 2, 2) == 2, "dupes 2nd");
        Check(dupes.Kth(0, 2, 3) == 2, "dupes 3rd");
        Check(dupes.Kth(1, 1, 1) == 2, "dupes single");

        var negatives = new PersistentKthSmallest(new[] { -3, 7, -3, 0 });
        Check(negatives.Kth(0, 3, 1) == -3, "neg 1st");
        Check(negatives.Kth(0, 3, 2) == -3, "neg 2nd");
        Check(negatives.Kth(0, 3, 3) == 0, "neg 3rd");
        Check(negatives.Kth(0, 3, 4) == 7, "neg 4th");
        Check(negatives.Kth(1, 2, 1) == -3, "neg slice 1st");
        Check(negatives.Kth(1, 2, 2) == 7, "neg slice 2nd");

        foreach (int config in new[] { 429, 12345, 54321 })
        {
            seed = config;
            int[] values = LcgArray(1 + seed % 40);
            var structure = new PersistentKthSmallest(values);
            int local = config;
            for (int step = 0; step < 40; step++)
            {
                local = (local * 75 + 74) % 65537;
                int from = local % values.Length;
                int to = local / 7 % values.Length;
                if (from > to) { int tmp = from; from = to; to = tmp; }
                var segment = new int[to - from + 1];
                Array.Copy(values, from, segment, 0, segment.Length);
                Array.Sort(segment);
                local = (local * 75 + 74) % 65537;
                int k = 1 + local % segment.Length;
                int got = structure.Kth(from, to, k);
                Check(got == segment[k - 1], "lcg query " + step + " got=" + got + " want=" + segment[k - 1]);
            }
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
