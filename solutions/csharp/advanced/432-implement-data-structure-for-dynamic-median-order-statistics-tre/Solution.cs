// Task 432 - Implement data structure for dynamic median (order-statistics tree). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Approach: two heaps with lazy deletion. Values up to the median live in a max-heap (low), the
// rest in a min-heap (high); every live low value is <= every live high value. Deletions do not
// search the heaps: a per-value pending count marks the removal and is consumed when a marked
// value surfaces at a heap top (prune), so remove/insert are O(log n) amortised. Median() returns
// the lower median - the live element at 0-based index (size-1)//2 - which is exactly the top of
// low, so even sizes answer with a stored value instead of a float; the empty multiset answers
// null. Cross-checked against a sorted-list oracle (binary-search insertion) on LCG operation
// scripts that insert, delete and re-query, including deletions until the structure empties.
using System;
using System.Collections.Generic;

internal class DynamicMedian
{
    // PriorityQueue<int, int> is a min-heap by priority; negate the priority to get a max-heap.
    private readonly PriorityQueue<int, int> low = new PriorityQueue<int, int>();
    private readonly PriorityQueue<int, int> high = new PriorityQueue<int, int>();
    private readonly Dictionary<int, int> pendingLow = new Dictionary<int, int>();
    private readonly Dictionary<int, int> pendingHigh = new Dictionary<int, int>();
    private readonly Dictionary<int, int> counts = new Dictionary<int, int>();
    private int total;
    private int lowCount;

    public int Size() => total;

    public int? Median()
    {
        if (total == 0) return null;
        Prune();
        return low.Peek();
    }

    public void Insert(int value)
    {
        Prune();
        counts[value] = counts.TryGetValue(value, out int seen) ? seen + 1 : 1;
        if (low.Count > 0 && value > low.Peek())
        {
            high.Enqueue(value, value);
        }
        else
        {
            low.Enqueue(value, -value);
            lowCount++;
        }
        total++;
        Rebalance();
    }

    public bool Remove(int value)
    {
        if (!counts.TryGetValue(value, out int live) || live == 0) return false;
        if (live == 1) counts.Remove(value);
        else counts[value] = live - 1;
        Prune();
        if (low.Count > 0 && value <= low.Peek())
        {
            pendingLow[value] = pendingLow.TryGetValue(value, out int marked) ? marked + 1 : 1;
            lowCount--;
        }
        else
        {
            pendingHigh[value] = pendingHigh.TryGetValue(value, out int marked) ? marked + 1 : 1;
        }
        total--;
        Rebalance();
        return true;
    }

    private void Prune()
    {
        int marks;
        while (low.Count > 0 && pendingLow.TryGetValue(low.Peek(), out marks) && marks > 0)
        {
            int top = low.Dequeue();
            pendingLow[top] = marks - 1;
        }
        while (high.Count > 0 && pendingHigh.TryGetValue(high.Peek(), out marks) && marks > 0)
        {
            int top = high.Dequeue();
            pendingHigh[top] = marks - 1;
        }
    }

    private void Rebalance()
    {
        int target = (total + 1) / 2;
        while (true)
        {
            Prune();
            if (lowCount > target)
            {
                int top = low.Dequeue();
                lowCount--;
                high.Enqueue(top, top);
                continue;
            }
            if (lowCount < target)
            {
                int top = high.Dequeue();
                lowCount++;
                low.Enqueue(top, -top);
                continue;
            }
            break;
        }
    }
}

public static class Solution
{
    private static int? OracleMedian(List<int> sortedValues)
    {
        if (sortedValues.Count == 0) return null;
        return sortedValues[(sortedValues.Count - 1) / 2];
    }

    private static void SortedInsert(List<int> sortedValues, int value)
    {
        int index = sortedValues.BinarySearch(value);
        if (index < 0) index = ~index;
        sortedValues.Insert(index, value);
    }

    public static void Main()
    {
        var tracker = new DynamicMedian();
        Check(tracker.Median() == null, "empty median");
        Check(!tracker.Remove(0), "remove from empty");

        var inserts = new[] { 5, 3, 8, 9 };
        var expected = new[] { 5, 3, 5, 5 };
        for (int i = 0; i < inserts.Length; i++)
        {
            tracker.Insert(inserts[i]);
            Check(tracker.Median() == expected[i], "hand median");
        }
        Check(tracker.Size() == 4, "hand size");
        Check(tracker.Remove(5), "remove 5");
        Check(tracker.Median() == 8, "median after removing 5");
        Check(tracker.Remove(3), "remove 3");
        Check(tracker.Median() == 8, "median after removing 3");
        Check(tracker.Remove(9), "remove 9");
        Check(tracker.Median() == 8, "median after removing 9");
        Check(tracker.Remove(8), "remove 8");
        Check(tracker.Median() == null, "empty again");
        Check(tracker.Size() == 0, "size back to 0");
        Check(!tracker.Remove(8), "remove missing");
        tracker.Insert(7);
        Check(tracker.Median() == 7, "reinsert after empty");
        Check(tracker.Remove(7), "remove last");
        Check(tracker.Median() == null, "empty after last");

        for (int i = 0; i < 3; i++)
        {
            tracker.Insert(3);
            Check(tracker.Median() == 3, "duplicates median");
        }
        Check(tracker.Remove(3) && tracker.Remove(3), "duplicate removal");
        Check(tracker.Median() == 3, "one duplicate left");
        Check(tracker.Remove(3), "remove last duplicate");
        Check(tracker.Median() == null, "duplicates drained");

        tracker = new DynamicMedian();
        var oracle = new List<int>();
        for (int value = 1; value <= 10; value++)
        {
            tracker.Insert(value);
            SortedInsert(oracle, value);
        }
        foreach (int value in new[] { 10, 1, 9, 2, 8, 3, 7, 4, 6, 5 })
        {
            Check(tracker.Remove(value), "scripted removal");
            oracle.Remove(value);
            Check(tracker.Median() == OracleMedian(oracle), "median while draining");
        }
        Check(tracker.Median() == null, "drained to empty");

        tracker = new DynamicMedian();
        oracle = new List<int>();
        int seed = 4320;
        for (int step = 0; step < 400; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 4;
            if (op <= 1 || oracle.Count == 0)
            {
                seed = (seed * 75 + 74) % 65537;
                int value = seed % 40 - 20;
                tracker.Insert(value);
                SortedInsert(oracle, value);
            }
            else if (op == 2)
            {
                seed = (seed * 75 + 74) % 65537;
                int index = seed % oracle.Count;
                int value = oracle[index];
                oracle.RemoveAt(index);
                Check(tracker.Remove(value), "random removal");
            }
            else
            {
                seed = (seed * 75 + 74) % 65537;
                int missing = 100 + seed % 10;
                Check(!tracker.Remove(missing), "random absent removal");
            }
            Check(tracker.Median() == OracleMedian(oracle), "random median step " + step);
            Check(tracker.Size() == oracle.Count, "random size step " + step);
        }
        Check(tracker.Size() > 0, "workload left values");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
