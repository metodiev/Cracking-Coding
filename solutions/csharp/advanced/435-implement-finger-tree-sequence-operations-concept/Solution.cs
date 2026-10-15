// Task 435 - Implement finger tree (sequence operations) concept. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Concept version of a finger tree: a sequence annotated by a monoid so that every subtree caches
// the annotation of the elements below it, which is what lets a real finger tree answer prefix and
// range measures in O(log n). The annotation monoid used here is M = (count, sum) with identity
// (0, 0) and componentwise addition; Measure() returns the cached annotation of the whole sequence
// and MeasureRange(lo, hi) folds it over a range. The spine operations PushFront / PushBack /
// PopFront / PopBack / Front / Back / Concat / SplitAt are simulated over a plain list while
// keeping the annotation exact - documented honest O(n) internals, no worst-case finger-tree bound
// claimed - and every scripted operation is asserted against a plain-list oracle (values, size,
// cached measure and the monoid law Combine(measure(a), measure(b)) == measure(Concat(a, b))).
using System;
using System.Collections.Generic;

internal class FingerTree
{
    private static readonly long[] Identity = { 0, 0 };

    private List<int> items;
    private long[] cached;

    public FingerTree() : this(new List<int>()) { }

    public FingerTree(List<int> initial)
    {
        items = new List<int>(initial);
        cached = (long[])Identity.Clone();
        foreach (int value in items) cached = Combine(cached, new long[] { 1, value });
    }

    public static long[] Combine(long[] left, long[] right)
    {
        return new long[] { left[0] + right[0], left[1] + right[1] };
    }

    public static bool SameMeasure(long[] actual, long count, long sum)
    {
        return actual[0] == count && actual[1] == sum;
    }

    public int Size() => items.Count;

    public long[] Measure() => (long[])cached.Clone();

    public long[] MeasureRange(int lo, int hi)
    {
        if (lo < 0 || hi > items.Count || lo > hi) throw new IndexOutOfRangeException("range out of bounds");
        long[] total = (long[])Identity.Clone();
        for (int i = lo; i < hi; i++) total = Combine(total, new long[] { 1, items[i] });
        return total;
    }

    public int? Front() => items.Count == 0 ? null : items[0];

    public int? Back() => items.Count == 0 ? null : items[items.Count - 1];

    public FingerTree PushFront(int value)
    {
        items.Insert(0, value);
        cached = Combine(new long[] { 1, value }, cached);
        return this;
    }

    public FingerTree PushBack(int value)
    {
        items.Add(value);
        cached = Combine(cached, new long[] { 1, value });
        return this;
    }

    public int? PopFront()
    {
        if (items.Count == 0) return null;
        int value = items[0];
        items.RemoveAt(0);
        cached = Combine(new long[] { -1, -value }, cached);
        return value;
    }

    public int? PopBack()
    {
        if (items.Count == 0) return null;
        int value = items[items.Count - 1];
        items.RemoveAt(items.Count - 1);
        cached = Combine(new long[] { -1, -value }, cached);
        return value;
    }

    public FingerTree Concat(FingerTree other)
    {
        var joined = new FingerTree(items);
        joined.items.AddRange(other.items);
        joined.cached = Combine(cached, other.cached);
        return joined;
    }

    public FingerTree[] SplitAt(int index)
    {
        if (index < 0 || index > items.Count) throw new IndexOutOfRangeException("split index out of bounds");
        var left = new FingerTree(items.GetRange(0, index));
        var right = new FingerTree(items.GetRange(index, items.Count - index));
        return new[] { left, right };
    }

    public List<int> ToList() => new List<int>(items);
}

public static class Solution
{
    private static void CheckMeasure(FingerTree tree, long count, long sum, string label)
    {
        Check(FingerTree.SameMeasure(tree.Measure(), count, sum), label);
    }

    public static void Main()
    {
        var empty = new FingerTree();
        Check(empty.Size() == 0, "empty size");
        CheckMeasure(empty, 0, 0, "empty measure");
        Check(empty.ToList().Count == 0, "empty list");
        Check(empty.Front() == null && empty.Back() == null, "empty ends");
        Check(empty.PopFront() == null && empty.PopBack() == null, "empty pops");

        var tree = new FingerTree();
        tree.PushBack(1).PushBack(2).PushBack(3).PushFront(0);
        Check(string.Join(",", tree.ToList()) == "0,1,2,3", "hand list");
        Check(tree.Size() == 4, "hand size");
        CheckMeasure(tree, 4, 6, "hand measure");
        Check(tree.Front() == 0 && tree.Back() == 3, "hand ends");

        var parts = tree.SplitAt(0);
        Check(parts[0].ToList().Count == 0 && string.Join(",", parts[1].ToList()) == "0,1,2,3", "split at 0");
        CheckMeasure(parts[0], 0, 0, "split 0 left measure");
        parts = tree.SplitAt(4);
        Check(string.Join(",", parts[0].ToList()) == "0,1,2,3" && parts[1].ToList().Count == 0, "split at len");
        CheckMeasure(parts[1], 0, 0, "split len right measure");
        parts = tree.SplitAt(2);
        Check(string.Join(",", parts[0].ToList()) == "0,1", "split middle left");
        Check(string.Join(",", parts[1].ToList()) == "2,3", "split middle right");
        Check(FingerTree.SameMeasure(FingerTree.Combine(parts[0].Measure(), parts[1].Measure()), 4, 6), "split measures combine");
        Check(FingerTree.SameMeasure(tree.MeasureRange(1, 3), 2, 3), "range measure");
        Check(FingerTree.SameMeasure(tree.MeasureRange(2, 2), 0, 0), "empty range measure");

        Check(string.Join(",", new FingerTree(new List<int> { 9 }).Concat(new FingerTree()).ToList()) == "9", "concat empty right");
        Check(string.Join(",", new FingerTree().Concat(new FingerTree(new List<int> { 7 })).ToList()) == "7", "concat empty left");
        CheckMeasure(new FingerTree(new List<int> { 1, 2 }).Concat(new FingerTree(new List<int> { 3 })), 3, 6, "concat measure");
        Check(string.Join(",", tree.Concat(tree).ToList()) == "0,1,2,3,0,1,2,3", "self concat");

        tree = new FingerTree(new List<int> { 5, 6, 7 });
        Check(tree.PopFront() == 5, "pop front value");
        Check(tree.PopBack() == 7, "pop back value");
        Check(string.Join(",", tree.ToList()) == "6", "single left");
        CheckMeasure(tree, 1, 6, "single measure");
        Check(tree.PopFront() == 6, "pop last");
        Check(tree.PopFront() == null, "pop empty");
        Check(tree.ToList().Count == 0 && tree.Measure()[0] == 0, "drained");

        tree = new FingerTree();
        var oracle = new List<int>();
        int seed = 4350;
        for (int step = 0; step < 400; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 5;
            if (op == 0)
            {
                seed = (seed * 75 + 74) % 65537;
                int value = seed % 50 - 25;
                tree.PushBack(value);
                oracle.Add(value);
            }
            else if (op == 1)
            {
                seed = (seed * 75 + 74) % 65537;
                int value = seed % 50 - 25;
                tree.PushFront(value);
                oracle.Insert(0, value);
            }
            else if (op == 2)
            {
                int? expected = oracle.Count == 0 ? (int?)null : oracle[0];
                if (oracle.Count > 0) oracle.RemoveAt(0);
                Check(tree.PopFront() == expected, "pop front step " + step);
            }
            else if (op == 3)
            {
                int? expected = oracle.Count == 0 ? (int?)null : oracle[oracle.Count - 1];
                if (oracle.Count > 0) oracle.RemoveAt(oracle.Count - 1);
                Check(tree.PopBack() == expected, "pop back step " + step);
            }
            else
            {
                seed = (seed * 75 + 74) % 65537;
                int index = seed % (oracle.Count + 1);
                var split = tree.SplitAt(index);
                Check(string.Join(",", split[0].ToList()) == string.Join(",", oracle.GetRange(0, index)), "split left step " + step);
                Check(string.Join(",", split[1].ToList()) == string.Join(",", oracle.GetRange(index, oracle.Count - index)), "split right step " + step);
                Check(FingerTree.SameMeasure(FingerTree.Combine(split[0].Measure(), split[1].Measure()),
                        tree.Measure()[0], tree.Measure()[1]), "split combine step " + step);
                tree = split[0].Concat(split[1]);
            }
            Check(string.Join(",", tree.ToList()) == string.Join(",", oracle), "list step " + step);
            Check(tree.Size() == oracle.Count, "size step " + step);
            long sum = 0;
            foreach (int value in oracle) sum += value;
            CheckMeasure(tree, oracle.Count, sum, "measure step " + step);
            if (oracle.Count > 0)
            {
                Check(tree.Front() == oracle[0] && tree.Back() == oracle[oracle.Count - 1], "ends step " + step);
            }
            seed = (seed * 75 + 74) % 65537;
            int lo = seed % (oracle.Count + 1);
            seed = (seed * 75 + 74) % 65537;
            int hi = lo + seed % (oracle.Count - lo + 1);
            long rangeSum = 0;
            for (int i = lo; i < hi; i++) rangeSum += oracle[i];
            Check(FingerTree.SameMeasure(tree.MeasureRange(lo, hi), hi - lo, rangeSum), "range step " + step);
        }
        Check(oracle.Count > 0, "workload left values");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
