// Task 445 - Implement fractional cascading for multiple binary searches. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;

internal class FractionalCascade
{
    private readonly int k;
    private readonly int[][] augmented;
    private readonly bool[][] nativeFlag;
    private readonly int[][] bridge;
    private readonly int[][] nextNative;

    public FractionalCascade(int[][] lists)
    {
        k = lists.Length;
        augmented = new int[k][];
        nativeFlag = new bool[k][];
        augmented[k - 1] = (int[])lists[k - 1].Clone();
        nativeFlag[k - 1] = new bool[augmented[k - 1].Length];
        for (int j = 0; j < nativeFlag[k - 1].Length; j++) nativeFlag[k - 1][j] = true;
        for (int i = k - 2; i >= 0; i--)
        {
            int[] next = augmented[i + 1];
            var copied = new int[(next.Length + 1) / 2];
            for (int j = 0; j < copied.Length; j++) copied[j] = next[2 * j];
            var values = new int[lists[i].Length + copied.Length];
            var flags = new bool[values.Length];
            int p = 0, q = 0, idx = 0;
            while (p < lists[i].Length || q < copied.Length)
            {
                if (q == copied.Length || (p < lists[i].Length && lists[i][p] <= copied[q]))
                {
                    values[idx] = lists[i][p++];
                    flags[idx++] = true;
                }
                else
                {
                    values[idx] = copied[q++];
                    flags[idx++] = false;
                }
            }
            augmented[i] = values;
            nativeFlag[i] = flags;
        }
        bridge = new int[k][];
        for (int i = 0; i < k - 1; i++)
        {
            int[] upper = augmented[i + 1];
            var b = new int[augmented[i].Length];
            int j = 0;
            for (int p = 0; p < augmented[i].Length; p++)
            {
                while (j < upper.Length && upper[j] < augmented[i][p]) j++;
                b[p] = j;
            }
            bridge[i] = b;
        }
        nextNative = new int[k][];
        for (int i = 0; i < k; i++)
        {
            int size = augmented[i].Length;
            var nxt = new int[size + 1];
            for (int p = 0; p <= size; p++) nxt[p] = size;
            for (int p = size - 1; p >= 0; p--) nxt[p] = nativeFlag[i][p] ? p : nxt[p + 1];
            nextNative[i] = nxt;
        }
    }

    public int?[] Successors(int x)
    {
        var results = new int?[k];
        int pos = BisectLeft(augmented[0], x);
        for (int i = 0; i < k; i++)
        {
            if (i > 0)
            {
                if (pos == augmented[i - 1].Length)
                {
                    int m = augmented[i].Length;
                    pos = m > 0 && augmented[i][m - 1] >= x ? m - 1 : m;
                }
                else
                {
                    int q = bridge[i - 1][pos];
                    if (q > 0 && augmented[i][q - 1] >= x) q--;
                    pos = q;
                }
            }
            int nxt = nextNative[i][pos];
            results[i] = nxt < augmented[i].Length ? augmented[i][nxt] : (int?)null;
        }
        return results;
    }

    private static int BisectLeft(int[] values, int x)
    {
        int lo = 0, hi = values.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (values[mid] < x) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}

public static class Solution
{
    private static int seed;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static int[][] LcgLists(int count, int size, int baseSeed)
    {
        seed = baseSeed;
        var lists = new int[count][];
        for (int i = 0; i < count; i++)
        {
            var values = new int[size];
            int current = -40;
            for (int j = 0; j < size; j++)
            {
                current += 1 + NextSeed() % 9;
                values[j] = current;
            }
            lists[i] = values;
        }
        return lists;
    }

    private static int?[] IndependentSuccessors(int[][] lists, int x)
    {
        var results = new int?[lists.Length];
        for (int i = 0; i < lists.Length; i++)
        {
            int pos = LowerBound(lists[i], x);
            results[i] = pos < lists[i].Length ? lists[i][pos] : (int?)null;
        }
        return results;
    }

    private static int LowerBound(int[] values, int x)
    {
        int lo = 0, hi = values.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (values[mid] < x) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private static bool Same(int?[] a, int?[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    public static void Main()
    {
        var cascade = new FractionalCascade(new int[][] { new int[] { 1, 10 }, new int[] { 5, 20 }, new int[] { 3, 7, 30 } });
        Check(Same(cascade.Successors(6), new int?[] { 10, 20, 7 }), "hand query 6");
        Check(Same(cascade.Successors(3), new int?[] { 10, 5, 3 }), "hand query 3");
        Check(Same(cascade.Successors(0), new int?[] { 1, 5, 3 }), "hand query 0");
        Check(Same(cascade.Successors(10), new int?[] { 10, 20, 30 }), "hand query 10");
        Check(Same(cascade.Successors(31), new int?[] { null, null, null }), "hand query past end");
        Check(Same(cascade.Successors(20), new int?[] { null, 20, 30 }), "hand query 20");

        int[][] lists = LcgLists(5, 12, 445);
        cascade = new FractionalCascade(lists);
        seed = 555;
        for (int step = 0; step < 200; step++)
        {
            int x = NextSeed() % 200 - 90;
            Check(Same(cascade.Successors(x), IndependentSuccessors(lists, x)), "random query " + x);
        }

        int[][] single = LcgLists(1, 7, 616);
        cascade = new FractionalCascade(single);
        seed = 77;
        for (int step = 0; step < 40; step++)
        {
            int x = NextSeed() % 120 - 60;
            Check(Same(cascade.Successors(x), IndependentSuccessors(single, x)), "single list query " + x);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
