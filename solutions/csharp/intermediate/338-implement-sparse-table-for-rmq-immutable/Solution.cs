// Task 338 - Implement sparse table for RMQ (immutable). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;

public class SparseTable
{
    // Row j holds the minimum of every window of length 2^j; a query combines the two
    // overlapping power-of-two windows that cover [left, right], giving O(n log n) build
    // and O(1) queries.
    private readonly int _n;
    private readonly int[] _logs;
    private readonly int[][] _table;

    public SparseTable(int[] values)
    {
        if (values.Length == 0) throw new Exception("values must not be empty");
        _n = values.Length;
        _logs = new int[_n + 1];
        for (int i = 2; i <= _n; i++) _logs[i] = _logs[i / 2] + 1;
        int levels = _logs[_n] + 1;
        _table = new int[levels][];
        _table[0] = (int[])values.Clone();
        for (int level = 1; level < levels; level++)
        {
            int half = 1 << (level - 1);
            int width = _n - (1 << level) + 1;
            _table[level] = new int[width];
            for (int i = 0; i < width; i++)
            {
                _table[level][i] = Math.Min(_table[level - 1][i], _table[level - 1][i + half]);
            }
        }
    }

    /// <summary>Minimum of values[left..right], both ends included.</summary>
    public int Query(int left, int right)
    {
        if (left < 0 || left > right || right >= _n) throw new Exception("invalid range");
        int level = _logs[right - left + 1];
        return Math.Min(_table[level][left], _table[level][right - (1 << level) + 1]);
    }
}

public static class Solution
{
    private static int seed = 8383;

    private static int NaiveMin(int[] values, int left, int right)
    {
        int best = values[left];
        for (int i = left + 1; i <= right; i++) best = Math.Min(best, values[i]);
        return best;
    }

    public static void Main()
    {
        // Hand-checked array; every expectation below is verified by eye.
        var values = new[] { 4, 2, 7, 1, 3, 9, 0, 5 };
        var table = new SparseTable(values);
        Check(table.Query(0, 7) == 0, "full range");
        Check(table.Query(1, 3) == 1, "1..3");
        Check(table.Query(0, 0) == 4, "0..0");
        Check(table.Query(1, 1) == 2, "1..1");
        Check(table.Query(2, 2) == 7, "2..2");
        Check(table.Query(3, 7) == 0, "3..7");
        Check(table.Query(4, 6) == 0, "4..6");
        Check(table.Query(2, 5) == 1, "2..5");
        Check(table.Query(5, 7) == 0, "5..7");
        Check(table.Query(0, 2) == 2, "0..2");
        Check(table.Query(6, 6) == 0, "6..6");

        var dup = new SparseTable(new[] { 5, 5, 5, 5 });
        Check(dup.Query(0, 3) == 5 && dup.Query(1, 2) == 5, "duplicates");
        var mixed = new SparseTable(new[] { 3, -1, 4, -1, 5, -9, 2, 6, -9, 7 });
        Check(mixed.Query(0, 9) == -9 && mixed.Query(5, 8) == -9 && mixed.Query(0, 3) == -1,
            "negatives and ties");
        var single = new SparseTable(new[] { 42 });
        Check(single.Query(0, 0) == 42, "single element");

        for (int trial = 0; trial < 8; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int size = 1 + seed % 64;
            var randomValues = new int[size];
            for (int i = 0; i < size; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                randomValues[i] = seed % 1000;
            }
            var randomTable = new SparseTable(randomValues);
            for (int left = 0; left < size; left++)
            {
                for (int right = left; right < size; right++)
                {
                    Check(randomTable.Query(left, right) == NaiveMin(randomValues, left, right),
                        "random " + trial + " [" + left + "," + right + "]");
                }
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
