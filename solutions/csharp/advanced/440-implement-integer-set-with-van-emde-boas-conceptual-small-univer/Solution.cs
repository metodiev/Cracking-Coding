// Task 440 - Implement integer set with van Emde Boas (conceptual small universe). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Documented simplification of a van Emde Boas set on the small universe U = 256 = 16 x 16: the top
// level keeps min/max, a 16-entry summary of cluster occupancy and 16 lazily created clusters, while
// the summary and every cluster is the base case - a 16-bit member bitmask (BitSet16) whose bit scans
// replace the recursion below u = 16. The decisive vEB trick is kept: the minimum is stored outside
// the clusters, so delete(min) promotes the successor from the cluster/summary structure. Operations
// MEMBER / SUCCESSOR / PREDECESSOR / INSERT / DELETE therefore match a full recursive vEB
// (O(1) bit work at this two-level scale). Assertions cover hand sequences that cross cluster
// boundaries, empty the structure and exercise successor/predecessor at both universe ends, plus LCG
// operation scripts cross-checked against a sorted-set oracle with full 0..255 sweeps.
using System;
using System.Collections.Generic;

internal class BitSet16
{
    private int mask;

    public void Insert(int value) => mask |= 1 << value;

    public void Remove(int value) => mask &= ~(1 << value);

    public bool Member(int value) => ((mask >> value) & 1) == 1;

    public bool Empty() => mask == 0;

    private static int LowestBit(int bits)
    {
        int index = 0;
        while ((bits & 1) == 0)
        {
            bits >>= 1;
            index++;
        }
        return index;
    }

    private static int HighestBit(int bits)
    {
        int index = -1;
        while (bits != 0)
        {
            bits >>= 1;
            index++;
        }
        return index;
    }

    public int? Minimum() => mask == 0 ? (int?)null : LowestBit(mask);

    public int? Maximum() => mask == 0 ? (int?)null : HighestBit(mask);

    public int? Successor(int value)
    {
        int higher = mask >> (value + 1);
        return higher == 0 ? (int?)null : value + 1 + LowestBit(higher);
    }

    public int? Predecessor(int value)
    {
        int lower = mask & ((1 << value) - 1);
        return lower == 0 ? (int?)null : HighestBit(lower);
    }
}

internal class VanEmdeBoas
{
    private int? min;
    private int? max;
    private int count;
    private BitSet16 summary = new BitSet16();
    private BitSet16?[] clusters = new BitSet16?[16];

    public int Size() => count;

    public int? Min => min;

    public int? Max => max;

    public bool Member(int value)
    {
        if (min != null && value == min) return true;
        if (max != null && value == max) return true;
        BitSet16? cluster = clusters[value >> 4];
        return cluster != null && cluster.Member(value & 15);
    }

    public void Insert(int value)
    {
        if (Member(value)) return;
        if (min == null)
        {
            min = value;
            max = value;
            count = 1;
            return;
        }
        if (value < min)
        {
            int previousMin = min.Value;
            min = value;
            value = previousMin;
        }
        if (value > max) max = value;
        int high = value >> 4;
        int low = value & 15;
        if (clusters[high] == null)
        {
            clusters[high] = new BitSet16();
            summary.Insert(high);
        }
        clusters[high]!.Insert(low);
        count++;
    }

    public bool Delete(int value)
    {
        if (!Member(value)) return false;
        if (count == 1)
        {
            min = null;
            max = null;
            summary = new BitSet16();
            clusters = new BitSet16?[16];
            count = 0;
            return true;
        }
        if (value == min)
        {
            int promoted = SuccessorStored(value);
            min = promoted;
            DeleteStored(promoted);
            return true;
        }
        DeleteStored(value);
        return true;
    }

    private int SuccessorStored(int value)
    {
        int high = value >> 4;
        int low = value & 15;
        BitSet16? cluster = clusters[high];
        if (cluster != null && low < 15)
        {
            int? within = cluster.Successor(low);
            if (within != null) return (high << 4) | within.Value;
        }
        int nextHigh = summary.Successor(high)!.Value;
        return (nextHigh << 4) | clusters[nextHigh]!.Minimum()!.Value;
    }

    private void DeleteStored(int value)
    {
        int high = value >> 4;
        int low = value & 15;
        clusters[high]!.Remove(low);
        if (clusters[high]!.Empty())
        {
            summary.Remove(high);
            clusters[high] = null;
        }
        count--;
        if (value == max)
        {
            int? top = summary.Maximum();
            max = top == null ? min : ((top.Value << 4) | clusters[top.Value]!.Maximum()!.Value);
        }
    }

    public int? Successor(int value)
    {
        if (min == null || value >= max) return null;
        if (value < min) return min;
        int high = value >> 4;
        int low = value & 15;
        BitSet16? cluster = clusters[high];
        if (cluster != null && low < 15)
        {
            int? within = cluster.Successor(low);
            if (within != null) return (high << 4) | within.Value;
        }
        int? nextHigh = summary.Successor(high);
        if (nextHigh == null) return null;
        return (nextHigh.Value << 4) | clusters[nextHigh.Value]!.Minimum()!.Value;
    }

    public int? Predecessor(int value)
    {
        if (min == null || value <= min) return null;
        if (value > max) return max;
        int high = value >> 4;
        int low = value & 15;
        BitSet16? cluster = clusters[high];
        if (cluster != null && low > 0)
        {
            int? within = cluster.Predecessor(low);
            if (within != null) return (high << 4) | within.Value;
        }
        int? previous = summary.Predecessor(high);
        if (previous == null) return min;
        return (previous.Value << 4) | clusters[previous.Value]!.Maximum()!.Value;
    }
}

public static class Solution
{
    private static void CheckEquals(int? actual, int? expected, string label)
    {
        Check(actual == expected, label + " got " + (actual == null ? "null" : actual.ToString()));
    }

    public static void Main()
    {
        var veb = new VanEmdeBoas();
        Check(veb.Size() == 0, "empty size");
        Check(!veb.Member(0), "empty member");
        CheckEquals(veb.Successor(0), null, "empty successor");
        CheckEquals(veb.Predecessor(255), null, "empty predecessor");
        Check(!veb.Delete(0), "empty delete");

        veb.Insert(200);
        Check(veb.Size() == 1, "single size");
        Check(veb.Member(200) && !veb.Member(199), "single member");
        CheckEquals(veb.Successor(200), null, "single successor of max");
        CheckEquals(veb.Successor(5), 200, "single successor below");
        CheckEquals(veb.Predecessor(200), null, "single predecessor of min");
        CheckEquals(veb.Predecessor(255), 200, "single predecessor above");
        Check(veb.Delete(200), "single delete");
        Check(!veb.Delete(200), "double delete");
        Check(veb.Size() == 0, "single drained");

        veb.Insert(5);
        veb.Insert(100);
        veb.Insert(200);
        Check(veb.Size() == 3, "hand size");
        CheckEquals(veb.Successor(5), 100, "hand successor middle");
        CheckEquals(veb.Successor(4), 5, "hand successor from below");
        CheckEquals(veb.Predecessor(100), 5, "hand predecessor middle");
        CheckEquals(veb.Predecessor(101), 100, "hand predecessor close");
        CheckEquals(veb.Successor(199), 200, "hand successor near max");
        Check(veb.Delete(100), "hand delete middle");
        Check(!veb.Member(100), "hand absent after delete");
        CheckEquals(veb.Successor(5), 200, "successor skips hole");
        CheckEquals(veb.Predecessor(200), 5, "predecessor skips hole");
        Check(veb.Delete(5), "delete min promotion");
        CheckEquals(veb.Successor(199), 200, "min promotion successor");
        CheckEquals(veb.Predecessor(199), null, "min promotion predecessor");
        Check(veb.Delete(200), "delete last");
        CheckEquals(veb.Successor(0), null, "empty again");

        veb.Insert(0);
        veb.Insert(15);
        veb.Insert(16);
        veb.Insert(17);
        veb.Insert(255);
        CheckEquals(veb.Successor(0), 15, "boundary 0->15");
        CheckEquals(veb.Successor(15), 16, "boundary cluster 0/1");
        CheckEquals(veb.Successor(16), 17, "boundary 16->17");
        CheckEquals(veb.Successor(17), 255, "boundary to 255");
        CheckEquals(veb.Predecessor(255), 17, "boundary from 255");
        CheckEquals(veb.Predecessor(17), 16, "boundary 17->16");
        CheckEquals(veb.Predecessor(16), 15, "boundary cluster 1/0");
        CheckEquals(veb.Predecessor(0), null, "boundary below 0");
        Check(veb.Delete(16), "boundary delete");
        CheckEquals(veb.Successor(15), 17, "after boundary delete");
        Check(veb.Delete(0), "delete min 0");
        Check(veb.Min == 15, "min promoted to 15");
        Check(veb.Delete(15), "delete min 15");
        Check(veb.Min == 17, "min promoted to 17");
        Check(veb.Delete(17), "delete min 17");
        Check(veb.Min == 255 && veb.Max == 255, "last element state");
        Check(veb.Delete(255), "delete 255");
        Check(veb.Size() == 0 && veb.Min == null, "final empty");

        veb = new VanEmdeBoas();
        var oracle = new List<int>();
        int seed = 4400;
        for (int step = 0; step < 600; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 5;
            seed = (seed * 75 + 74) % 65537;
            int value = seed % 256;
            if (op <= 1)
            {
                veb.Insert(value);
                int index = oracle.BinarySearch(value);
                if (index < 0) oracle.Insert(~index, value);
            }
            else if (op == 2)
            {
                bool expected = oracle.Contains(value);
                if (expected) oracle.Remove(value);
                Check(veb.Delete(value) == expected, "random delete step " + step);
            }
            else if (op == 3)
            {
                Check(veb.Member(value) == oracle.Contains(value), "random member step " + step);
            }
            else
            {
                int? expectedSuccessor = null;
                int? expectedPredecessor = null;
                foreach (int candidate in oracle)
                {
                    if (candidate > value && expectedSuccessor == null) expectedSuccessor = candidate;
                    if (candidate < value) expectedPredecessor = candidate;
                }
                CheckEquals(veb.Successor(value), expectedSuccessor, "random successor step " + step);
                CheckEquals(veb.Predecessor(value), expectedPredecessor, "random predecessor step " + step);
            }
            Check(veb.Size() == oracle.Count, "random size step " + step);
            if (step % 50 == 0)
            {
                for (int probe = 0; probe < 256; probe++)
                {
                    Check(veb.Member(probe) == oracle.Contains(probe), "sweep member step " + step + " probe " + probe);
                }
            }
        }
        Check(oracle.Count > 0, "workload left values");
        for (int probe = 0; probe < 256; probe++)
        {
            Check(veb.Member(probe) == oracle.Contains(probe), "final member probe " + probe);
            int? expectedSuccessor = null;
            int? expectedPredecessor = null;
            foreach (int candidate in oracle)
            {
                if (candidate > probe && expectedSuccessor == null) expectedSuccessor = candidate;
                if (candidate < probe) expectedPredecessor = candidate;
            }
            CheckEquals(veb.Successor(probe), expectedSuccessor, "final successor probe " + probe);
            CheckEquals(veb.Predecessor(probe), expectedPredecessor, "final predecessor probe " + probe);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
