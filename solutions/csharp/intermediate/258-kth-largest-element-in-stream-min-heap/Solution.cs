// Task 258 - Kth largest element in stream (min-heap). (Tier: Intermediate, CTCI Ch 10)
using System;
using System.Collections.Generic;

public class KthLargest
{
    private readonly int _k;
    private readonly PriorityQueue<int, int> _heap = new PriorityQueue<int, int>();

    // Keeps only the k largest values seen, so Peek() is the k-th largest.
    // Add() returns null until at least k values have been seen.
    public KthLargest(int k, int[] nums)
    {
        if (k < 1) throw new Exception("k must be >= 1");
        _k = k;
        foreach (int v in nums) _heap.Enqueue(v, v);
        while (_heap.Count > k) _heap.Dequeue();
    }

    public int? Add(int val)
    {
        _heap.Enqueue(val, val);
        if (_heap.Count > _k) _heap.Dequeue();
        if (_heap.Count < _k) return null;
        return _heap.Peek();
    }
}

public static class Solution
{
    public static void Main()
    {
        var kl = new KthLargest(3, new[] { 4, 5, 8, 2 });
        Check(kl.Add(3) == 4, "add 3 -> 4");
        Check(kl.Add(5) == 5, "add 5 -> 5");
        Check(kl.Add(10) == 5, "add 10 -> 5");
        Check(kl.Add(9) == 8, "add 9 -> 8");
        Check(kl.Add(4) == 8, "add 4 -> 8");

        var grow = new KthLargest(5, new[] { 3, 1 });        // k larger than the initial count
        Check(grow.Add(2) == null, "3 values seen: still waiting");
        Check(grow.Add(7) == null, "4 values seen: still waiting");
        Check(grow.Add(9) == 1, "5th largest of {1,2,3,7,9}");
        Check(grow.Add(4) == 2, "5th largest of {2,3,4,7,9}");

        var top = new KthLargest(1, new[] { 4, 5, 8, 2 });   // k = 1: running maximum
        Check(top.Add(3) == 8, "max ignores 3");
        Check(top.Add(10) == 10, "new max");
        Check(top.Add(4) == 10, "max stays");

        var dupes = new KthLargest(2, new[] { 5, 5, 5 });
        Check(dupes.Add(5) == 5, "duplicates all");
        Check(dupes.Add(4) == 5, "4 is too small");
        Check(dupes.Add(6) == 5, "{5,5,6}");
        Check(dupes.Add(7) == 6, "{6,7} keeps 2nd largest");

        var empty = new KthLargest(2, new int[0]);
        Check(empty.Add(5) == null, "empty stream: waiting");
        Check(empty.Add(3) == 3, "{3,5} -> 3");
        Check(empty.Add(10) == 5, "{5,10} -> 5");
        Check(empty.Add(8) == 8, "{8,10} -> 8");

        var start = new KthLargest(2, new[] { 1, 2, 3 });    // initial pile trimmed to top k
        Check(start.Add(0) == 2, "top 2 of {0,2,3} -> 2");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
