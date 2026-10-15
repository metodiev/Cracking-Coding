// Task 336 - Design a snapshot array (persisting versions basic). (Tier: Intermediate, CTCI Ch 7)
using System;
using System.Collections.Generic;

public class SnapshotArray
{
    // history[i] holds [snapId, value] pairs with strictly increasing snap ids; the entry for
    // snap 0 defaults to 0. Get() binary-searches the last write with snapId <= requested.
    private readonly int _length;
    private readonly List<List<int[]>> _history = new List<List<int[]>>();
    private int _snapCount = 0;

    public SnapshotArray(int length)
    {
        if (length < 1) throw new Exception("length must be >= 1");
        _length = length;
        for (int i = 0; i < length; i++)
        {
            _history.Add(new List<int[]> { new[] { 0, 0 } });
        }
    }

    public void Set(int index, int value)
    {
        if (index < 0 || index >= _length) throw new Exception("index out of range");
        List<int[]> entries = _history[index];
        int[] last = entries[entries.Count - 1];
        if (last[0] == _snapCount)
        {
            last[1] = value;                       // overwrite the current (unsealed) write
        }
        else
        {
            entries.Add(new[] { _snapCount, value });
        }
    }

    public int Snap()
    {
        return _snapCount++;
    }

    public int Get(int index, int snapId)
    {
        if (index < 0 || index >= _length) throw new Exception("index out of range");
        if (snapId < 0 || snapId >= _snapCount) throw new Exception("unknown snapshot id");
        List<int[]> entries = _history[index];
        int lo = 0, hi = entries.Count - 1;
        while (lo < hi)                            // last entry with snapId <= requested
        {
            int mid = (lo + hi + 1) / 2;
            if (entries[mid][0] <= snapId) lo = mid;
            else hi = mid - 1;
        }
        return entries[lo][1];
    }
}

public class NaiveSnapshotArray
{
    private readonly int[] _current;
    private readonly List<int[]> _versions = new List<int[]>();

    public NaiveSnapshotArray(int length)
    {
        _current = new int[length];
    }

    public void Set(int index, int value)
    {
        _current[index] = value;
    }

    public int Snap()
    {
        _versions.Add((int[])_current.Clone());
        return _versions.Count - 1;
    }

    public int Get(int index, int snapId)
    {
        return _versions[snapId][index];
    }

    public int VersionCount
    {
        get { return _versions.Count; }
    }
}

public static class Solution
{
    private static int seed = 3141;

    public static void Main()
    {
        // Canonical LeetCode sequence.
        var arr = new SnapshotArray(3);
        arr.Set(0, 5);
        Check(arr.Snap() == 0, "first snap id");
        arr.Set(0, 6);
        Check(arr.Get(0, 0) == 5, "old value survives");
        Check(arr.Snap() == 1, "second snap id");
        Check(arr.Get(0, 0) == 5, "snapshot 0 value");
        Check(arr.Get(0, 1) == 6, "snapshot 1 value");
        Check(arr.Get(1, 0) == 0 && arr.Get(2, 1) == 0, "untouched indices are zero");

        // Writes before the first snap are visible in snapshot 0; later writes are not.
        var arr2 = new SnapshotArray(2);
        arr2.Set(1, 7);
        arr2.Set(1, 9);                            // same snapshot 0: overwrites 7
        Check(arr2.Snap() == 0, "snap 0");
        Check(arr2.Get(1, 0) == 9, "overwrite before snap");
        arr2.Set(1, 11);
        Check(arr2.Get(1, 0) == 9, "unsealed write invisible");
        Check(arr2.Snap() == 1, "snap 1");
        Check(arr2.Get(1, 1) == 11, "new value visible in snapshot 1");

        // Snapshots with no writes in between repeat the previous value.
        var arr3 = new SnapshotArray(1);
        arr3.Set(0, 1);
        int first = arr3.Snap();
        int second = arr3.Snap();
        arr3.Set(0, 2);
        int third = arr3.Snap();
        Check(first == 0 && second == 1 && third == 2, "snap ids");
        Check(arr3.Get(0, 0) == 1 && arr3.Get(0, 1) == 1 && arr3.Get(0, 2) == 2, "repeated values");

        bool threw = false;
        try
        {
            arr3.Get(0, 3);
        }
        catch (Exception expected)
        {
            threw = true;
        }
        Check(threw, "unknown snapshot raises");

        // LCG operation sequences against the full-copy oracle.
        int length = 8;
        var fast = new SnapshotArray(length);
        var naive = new NaiveSnapshotArray(length);
        for (int step = 0; step < 3000; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            int choice = seed % 10;
            seed = (seed * 75 + 74) % 65537;
            if (choice < 5)
            {
                int index = seed % length;
                seed = (seed * 75 + 74) % 65537;
                int value = seed % 1000;
                fast.Set(index, value);
                naive.Set(index, value);
            }
            else if (choice < 7)
            {
                Check(fast.Snap() == naive.Snap(), "random snap " + step);
            }
            else if (naive.VersionCount > 0)
            {
                int index = seed % length;
                seed = (seed * 75 + 74) % 65537;
                int snapId = seed % naive.VersionCount;
                Check(fast.Get(index, snapId) == naive.Get(index, snapId), "random get " + step);
            }
            if (naive.VersionCount > 0)
            {
                int latest = naive.VersionCount - 1;
                for (int probe = 0; probe < length; probe++)
                {
                    Check(fast.Get(probe, latest) == naive.Get(probe, latest), "probe " + step);
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
