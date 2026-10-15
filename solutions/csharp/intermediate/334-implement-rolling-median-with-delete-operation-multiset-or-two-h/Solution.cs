// Task 334 - Implement rolling median with delete operation (multiset or two heaps + lazy deletion). (Tier: Intermediate, CTCI Ch 7)
using System;
using System.Collections.Generic;

public class MedianMultiset
{
    // `low` keeps the smaller half as a max-heap, `high` the larger half as a min-heap, so
    // counts satisfy low == high or low == high + 1. delete() marks one occurrence as pending
    // in whichever heap holds it; the mark is consumed when that value reaches a heap top.
    private readonly PriorityQueue<int, int> _low = new PriorityQueue<int, int>();
    private readonly PriorityQueue<int, int> _high = new PriorityQueue<int, int>();
    private readonly Dictionary<int, int> _pendingLow = new Dictionary<int, int>();
    private readonly Dictionary<int, int> _pendingHigh = new Dictionary<int, int>();
    private readonly Dictionary<int, int> _active = new Dictionary<int, int>();
    private int _pendingLowTotal = 0;
    private int _pendingHighTotal = 0;
    private int _count = 0;

    public int Size()
    {
        return _count;
    }

    public void Add(int value)
    {
        CleanLow();
        CleanHigh();
        if (_low.Count > 0 && value <= _low.Peek()) _low.Enqueue(value, -value);
        else _high.Enqueue(value, value);
        _count++;
        _active[value] = Count(_active, value) + 1;
        Rebalance();
    }

    /// <summary>Removes one occurrence of `value`; throws when it is absent.</summary>
    public void Delete(int value)
    {
        if (Count(_active, value) < 1) throw new Exception("value not present: " + value);
        _active[value] = _active[value] - 1;
        CleanLow();
        if (_low.Count > 0 && value <= _low.Peek())
        {
            _pendingLow[value] = Count(_pendingLow, value) + 1;
            _pendingLowTotal++;
        }
        else
        {
            _pendingHigh[value] = Count(_pendingHigh, value) + 1;
            _pendingHighTotal++;
        }
        _count--;
        CleanLow();
        CleanHigh();
        Rebalance();
    }

    /// <summary>Median as a double, or null when the multiset is empty.</summary>
    public double? Median()
    {
        CleanLow();
        CleanHigh();
        if (_count == 0) return null;
        if (_count % 2 == 1) return (double)_low.Peek();
        return (_low.Peek() + _high.Peek()) / 2.0;
    }

    private static int Count(Dictionary<int, int> map, int key)
    {
        return map.TryGetValue(key, out int value) ? value : 0;
    }

    private void CleanLow()
    {
        while (_low.Count > 0 && Count(_pendingLow, _low.Peek()) > 0)
        {
            int value = _low.Dequeue();
            _pendingLow[value] = _pendingLow[value] - 1;
            _pendingLowTotal--;
        }
    }

    private void CleanHigh()
    {
        while (_high.Count > 0 && Count(_pendingHigh, _high.Peek()) > 0)
        {
            int value = _high.Dequeue();
            _pendingHigh[value] = _pendingHigh[value] - 1;
            _pendingHighTotal--;
        }
    }

    private void Rebalance()
    {
        while (true)
        {
            int lowSize = _low.Count - _pendingLowTotal;
            int highSize = _high.Count - _pendingHighTotal;
            if (lowSize > highSize + 1)
            {
                CleanLow();
                int value = _low.Dequeue();
                _high.Enqueue(value, value);
            }
            else if (highSize > lowSize)
            {
                CleanHigh();
                int value = _high.Dequeue();
                _low.Enqueue(value, -value);
            }
            else
            {
                return;
            }
        }
    }
}

public class SortedListOracle
{
    public readonly List<int> Data = new List<int>();

    public void Add(int value)
    {
        int index = Data.BinarySearch(value);
        if (index < 0) index = ~index;
        Data.Insert(index, value);
    }

    public void Delete(int value)
    {
        int index = Data.BinarySearch(value);
        if (index < 0 || Data[index] != value) throw new Exception("oracle delete of absent value");
        Data.RemoveAt(index);
    }

    public double? Median()
    {
        int n = Data.Count;
        if (n == 0) return null;
        if (n % 2 == 1) return (double)Data[n / 2];
        return (Data[n / 2 - 1] + Data[n / 2]) / 2.0;
    }
}

public static class Solution
{
    private static int seed = 4242;

    public static void Main()
    {
        var m = new MedianMultiset();
        Check(m.Size() == 0 && m.Median() == null, "empty");
        m.Add(5);
        Check(m.Median() == 5.0, "median 5");
        m.Add(3);
        Check(m.Size() == 2 && m.Median() == 4.0, "median 4");
        m.Add(8);
        Check(m.Median() == 5.0, "median 5 again");
        m.Add(1);
        Check(m.Median() == 4.0, "median of {1,3,5,8}");
        m.Delete(5);
        Check(m.Size() == 3 && m.Median() == 3.0, "median of {1,3,8}");
        m.Delete(1);
        Check(m.Median() == 5.5, "median of {3,8}");
        m.Delete(3);
        Check(m.Median() == 8.0, "median of {8}");
        m.Delete(8);
        Check(m.Size() == 0 && m.Median() == null, "back to empty");
        bool threw = false;
        try
        {
            m.Delete(7);
        }
        catch (Exception)
        {
            threw = true;
        }
        Check(threw, "delete absent raises");

        var dupes = new MedianMultiset();
        for (int i = 0; i < 3; i++) dupes.Add(4);
        Check(dupes.Median() == 4.0, "dupes median");
        dupes.Delete(4);
        Check(dupes.Median() == 4.0, "dupes median after one delete");
        dupes.Delete(4);
        Check(dupes.Median() == 4.0 && dupes.Size() == 1, "dupes median after two deletes");
        dupes.Delete(4);
        Check(dupes.Median() == null, "dupes empty");

        var seq = new MedianMultiset();
        for (int value = 1; value <= 10; value++) seq.Add(value);
        Check(seq.Median() == 5.5, "seq median");
        var steps = new[] { 5, 6, 10, 1, 7, 2 };
        var expected = new[] { 6.0, 5.5, 4.0, 5.5, 4.0, 6.0 };
        for (int i = 0; i < steps.Length; i++)
        {
            seq.Delete(steps[i]);
            Check(seq.Median() == expected[i], "seq step " + i);
        }
        Check(seq.Size() == 4, "seq size");

        var fast = new MedianMultiset();
        var oracle = new SortedListOracle();
        for (int step = 0; step < 2000; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            if (oracle.Data.Count > 0 && seed % 10 >= 6)
            {
                seed = (seed * 75 + 74) % 65537;
                int value = oracle.Data[seed % oracle.Data.Count];
                fast.Delete(value);
                oracle.Delete(value);
            }
            else
            {
                seed = (seed * 75 + 74) % 65537;
                int value = 1 + seed % 50;
                fast.Add(value);
                oracle.Add(value);
            }
            Check(fast.Size() == oracle.Data.Count, "random size " + step);
            seed = (seed * 75 + 74) % 65537;
            if (seed % 5 == 0 || step % 7 == 0)
            {
                Check(fast.Median() == oracle.Median(), "random median " + step);
            }
        }
        while (oracle.Data.Count > 0)
        {
            int value = oracle.Data[0];
            fast.Delete(value);
            oracle.Delete(value);
            Check(fast.Median() == oracle.Median(), "drain median");
        }
        Check(fast.Median() == null, "drained empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
