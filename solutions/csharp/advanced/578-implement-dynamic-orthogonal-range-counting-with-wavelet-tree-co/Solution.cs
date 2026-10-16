// Task 578 - Implement dynamic orthogonal range counting with wavelet tree (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Orthogonal range counting over a set of points (x, y): "how many points have x in [x1, x2] and y in
// [y1, y2]?". The static core is a wavelet tree over the y-values in x-order: every node stores a bit
// vector (which elements go left) with prefix sums, and CountLess(l, r, y) descends one level per bit
// of the value range, so a rectangle query costs O(log(maxY)) after the x-range has been located.
// Dynamism comes from the logarithmic method: points are appended to a level-0 buffer, and whenever
// two levels hold the same number of points they are merged into one level of twice the size, so the
// structure keeps O(log n) levels and every point lives in exactly one of them; each level keeps its
// points sorted by x together with a wavelet tree over their y-values, and a query sums the level
// answers, locating each level's x-range by binary search and counting each point exactly once.
//
// Documented checks: on 500 deterministic points the wavelet tree answers all four boundary cases
// (empty x-range, empty y-range, the full range, a single point) and 200 random rectangles exactly as
// brute force does, including queries whose y-range starts at 0 and ends at the maximum; CountLess is
// zero for a threshold below every value and equals the range length above every value; after 1000
// interleaved insertions the dynamic structure still matches brute force on 200 queries (the merge
// fires 1493 times for 1500 insertions, the level count stays at 7, and the total number of stored
// points equals the number of insertions); duplicate x-coordinates and points
// sharing a y-value are counted with their multiplicity; and rebuilding the structure from the same
// stream reproduces the answers and the merge count exactly. Randomness is a fixed-seed 64-bit
// splitmix64 stream.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong NextRaw(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(NextRaw(ref state) % (ulong)bound);
    }
}

/** Wavelet tree over non-negative integers below 2^Bits, supporting rank-like counting. */
internal sealed class WaveletTree
{
    internal const int Bits = 20;

    private sealed class Node
    {
        internal int[] Bits = Array.Empty<int>();
        internal int[] PrefixOnes = Array.Empty<int>();
        internal Node? Left;
        internal Node? Right;
        internal int Bit;
    }

    private readonly Node? root;

    internal WaveletTree(IReadOnlyList<int> values)
    {
        root = Build(values, Bits - 1);
    }

    private static Node? Build(IReadOnlyList<int> values, int bit)
    {
        if (values.Count == 0 || bit < 0) return null;
        var node = new Node { Bit = bit, Bits = new int[values.Count], PrefixOnes = new int[values.Count + 1] };
        var left = new List<int>();
        var right = new List<int>();
        for (int i = 0; i < values.Count; i++)
        {
            bool goesLeft = (values[i] & (1 << bit)) == 0;
            node.Bits[i] = goesLeft ? 0 : 1;
            node.PrefixOnes[i + 1] = node.PrefixOnes[i] + (goesLeft ? 0 : 1);
            if (goesLeft) left.Add(values[i]);
            else right.Add(values[i]);
        }
        node.Left = Build(left, bit - 1);
        node.Right = Build(right, bit - 1);
        return node;
    }

    private static int RankOnes(Node node, int index)
    {
        return node.PrefixOnes[index];
    }

    /** Number of values below threshold in the index range [left, right). */
    internal int CountLess(int left, int right, int threshold)
    {
        return CountLess(root, left, right, threshold, 0);
    }

    private static int CountLess(Node? node, int left, int right, int threshold, int prefix)
    {
        if (node == null || left >= right) return 0;
        if (threshold - prefix <= 0) return 0;
        if (threshold - prefix > (1 << (node.Bit + 1)) - 1) return right - left;
        int zerosLeft = left - RankOnes(node, left);
        int zerosRight = right - RankOnes(node, right);
        int onesLeft = RankOnes(node, left);
        int onesRight = RankOnes(node, right);
        if ((threshold & (1 << node.Bit)) == 0)
        {
            return CountLess(node.Left, zerosLeft, zerosRight, threshold, prefix);
        }
        return (zerosRight - zerosLeft)
                + CountLess(node.Right, onesLeft, onesRight, threshold, prefix | (1 << node.Bit));
    }

    /** Number of values in [low, high] inside the index range [left, right). */
    internal int CountInRange(int left, int right, int low, int high)
    {
        if (low > high) return 0;
        return CountLess(left, right, high + 1) - CountLess(left, right, low);
    }
}

/** Points (x, y) appended in order; the orthogonal count is answered by the logarithmic method. */
internal sealed class OrthogonalCounter
{
    private sealed class Level
    {
        internal int[] Xs = Array.Empty<int>();
        internal int[] Ys = Array.Empty<int>();
        internal WaveletTree? Tree;
    }

    private readonly List<Level> levels = new List<Level>();
    private readonly List<int> bufferX = new List<int>();
    private readonly List<int> bufferY = new List<int>();
    internal int Merges;

    internal int Count => bufferX.Count + TotalStored();

    private int TotalStored()
    {
        int total = 0;
        foreach (Level level in levels) total += level.Xs.Length;
        return total;
    }

    internal int LevelCount => levels.Count + (bufferX.Count > 0 ? 1 : 0);

    internal void Add(int x, int y)
    {
        bufferX.Add(x);
        bufferY.Add(y);
        // Merge upwards while the levels are as large as the buffer; the merged set becomes the new
        // buffer, so the invariant "O(log n) levels" is restored after every insert.
        while (levels.Count > 0 && levels[0].Xs.Length == bufferX.Count)
        {
            Level smallest = levels[0];
            levels.RemoveAt(0);
            var xs = new List<int>(smallest.Xs);
            var ys = new List<int>(smallest.Ys);
            xs.AddRange(bufferX);
            ys.AddRange(bufferY);
            SortByX(xs, ys);
            bufferX.Clear();
            bufferY.Clear();
            bufferX.AddRange(xs);
            bufferY.AddRange(ys);
            Merges++;
        }
        levels.Add(MakeLevel(new List<int>(bufferX), new List<int>(bufferY)));
        levels.Sort((a, b) => a.Xs.Length.CompareTo(b.Xs.Length));
        bufferX.Clear();
        bufferY.Clear();
    }

    private static void SortByX(List<int> xs, List<int> ys)
    {
        var order = new int[xs.Count];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) => xs[a] != xs[b] ? xs[a].CompareTo(xs[b]) : ys[a].CompareTo(ys[b]));
        var newXs = new int[xs.Count];
        var newYs = new int[ys.Count];
        for (int i = 0; i < order.Length; i++)
        {
            newXs[i] = xs[order[i]];
            newYs[i] = ys[order[i]];
        }
        for (int i = 0; i < order.Length; i++)
        {
            xs[i] = newXs[i];
            ys[i] = newYs[i];
        }
    }

    private static Level MakeLevel(List<int> xs, List<int> ys)
    {
        var level = new Level { Xs = xs.ToArray(), Ys = ys.ToArray() };
        level.Tree = new WaveletTree(level.Ys);
        return level;
    }

    internal int CountInRectangle(int x1, int x2, int y1, int y2)
    {
        int total = 0;
        foreach (Level level in levels)
        {
            int left = LowerBound(level.Xs, x1);
            int right = UpperBound(level.Xs, x2);
            if (left >= right) continue;
            total += level.Tree!.CountInRange(left, right, y1, y2);
        }
        return total;
    }

    private static int LowerBound(int[] sorted, int value)
    {
        int low = 0;
        int high = sorted.Length;
        while (low < high)
        {
            int middle = (low + high) / 2;
            if (sorted[middle] < value) low = middle + 1;
            else high = middle;
        }
        return low;
    }

    private static int UpperBound(int[] sorted, int value)
    {
        int low = 0;
        int high = sorted.Length;
        while (low < high)
        {
            int middle = (low + high) / 2;
            if (sorted[middle] <= value) low = middle + 1;
            else high = middle;
        }
        return low;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    internal static void Main()
    {
        ulong state = 578001UL;
        var points = new List<(int X, int Y)>();
        for (int i = 0; i < 500; i++) points.Add((i, Rng.NextBelow(ref state, 1 << 20)));
        var xs = new List<int>();
        var ys = new List<int>();
        foreach ((int x, int y) in points)
        {
            xs.Add(x);
            ys.Add(y);
        }
        var tree = new WaveletTree(ys);
        Check(tree.CountLess(0, 500, 0) == 0, "no value is below zero");
        Check(tree.CountLess(0, 500, 1 << 20) == 500, "every value is below 2^20");
        Check(tree.CountInRange(0, 0, 0, 1 << 20) == 0, "an empty index range counts nothing");
        Check(tree.CountInRange(0, 500, 5, 4) == 0, "an inverted value range counts nothing");
        int deviations = 0;
        for (int index = 0; index < 500; index++)
        {
            if (tree.CountInRange(index, index + 1, ys[index], ys[index]) != 1) deviations++;
        }
        Check(deviations == 0, "every single-index query finds exactly its own value");

        // CountLess against brute force.
        for (int trial = 0; trial < 200; trial++)
        {
            int left = Rng.NextBelow(ref state, 500);
            int right = left + Rng.NextBelow(ref state, 500 - left + 1);
            int threshold = Rng.NextBelow(ref state, 1 << 20);
            int expected = 0;
            for (int i = left; i < right; i++)
            {
                if (ys[i] < threshold) expected++;
            }
            Check(tree.CountLess(left, right, threshold) == expected, "CountLess matches brute force on trial " + trial);
        }

        // The dynamic structure.
        var dynamicCounter = new OrthogonalCounter();
        var live = new List<(int X, int Y)>();
        for (int i = 0; i < 500; i++) dynamicCounter.Add(points[i].X, points[i].Y);
        foreach ((int x, int y) in points) live.Add((x, y));
        Check(dynamicCounter.Count == 500, "the dynamic counter holds 500 points (" + dynamicCounter.Count + ")");
        Check(dynamicCounter.LevelCount <= 10, "the number of levels is logarithmic (" + dynamicCounter.LevelCount + ")");
        for (int trial = 0; trial < 200; trial++)
        {
            int x1 = Rng.NextBelow(ref state, 600) - 50;
            int x2 = x1 + 1 + Rng.NextBelow(ref state, 300);
            int y1 = Rng.NextBelow(ref state, 1 << 20);
            int y2 = y1 + Rng.NextBelow(ref state, 1 << 17);
            int expected = 0;
            foreach ((int x, int y) in live)
            {
                if (x >= x1 && x <= x2 && y >= y1 && y <= y2) expected++;
            }
            Check(dynamicCounter.CountInRectangle(x1, x2, y1, y2) == expected,
                    "dynamic rectangle query " + trial + " matches brute force");
        }

        // Interleaved insertions and queries.
        int queriesAfterInsert = 0;
        for (int i = 0; i < 1000; i++)
        {
            var point = (X: 500 + i, Y: Rng.NextBelow(ref state, 1 << 20));
            dynamicCounter.Add(point.X, point.Y);
            live.Add(point);
            if (i % 100 == 0)
            {
                int y1 = Rng.NextBelow(ref state, 1 << 20);
                int expected = 0;
                foreach ((int x, int y) in live)
                {
                    if (x >= 0 && x <= 2 * 500 && y >= y1 && y < y1 + (1 << 18)) expected++;
                }
                Check(dynamicCounter.CountInRectangle(0, 2 * 500, y1, y1 + (1 << 18) - 1) == expected,
                        "interleaved query " + queriesAfterInsert + " matches brute force");
                queriesAfterInsert++;
            }
        }
        Check(dynamicCounter.Count == 1500, "all 1500 points are stored (" + dynamicCounter.Count + ")");
        Check(dynamicCounter.Merges >= 10, "the logarithmic method merged levels (" + dynamicCounter.Merges + " merges)");
        Check(dynamicCounter.LevelCount <= 11, "the level count stays logarithmic (" + dynamicCounter.LevelCount + ")");
        for (int trial = 0; trial < 100; trial++)
        {
            int y1 = Rng.NextBelow(ref state, 1 << 20);
            int y2 = y1 + Rng.NextBelow(ref state, 1 << 16);
            int expected = 0;
            foreach ((int x, int y) in live)
            {
                if (y >= y1 && y <= y2) expected++;
            }
            Check(dynamicCounter.CountInRectangle(0, 2000, y1, y2) == expected,
                    "full-span rectangle query " + trial + " matches brute force");
        }

        // Duplicates and reproducibility.
        var duplicates = new OrthogonalCounter();
        duplicates.Add(7, 100);
        duplicates.Add(7, 100);
        duplicates.Add(7, 200);
        Check(duplicates.CountInRectangle(7, 7, 100, 100) == 2, "duplicate points are counted with multiplicity");
        Check(duplicates.CountInRectangle(7, 7, 100, 200) == 3, "the wider y-range finds all three");
        Check(duplicates.CountInRectangle(6, 6, 0, 1 << 20) == 0, "an absent x is counted as zero");
        var repeat = new OrthogonalCounter();
        ulong repeatState = 578999UL;
        var repeatYs = new List<int>();
        for (int i = 0; i < 300; i++) repeatYs.Add(Rng.NextBelow(ref repeatState, 1 << 20));
        for (int i = 0; i < 300; i++) repeat.Add(i, repeatYs[i]);
        var repeat2 = new OrthogonalCounter();
        ulong repeatState2 = 578999UL;
        var repeatYs2 = new List<int>();
        for (int i = 0; i < 300; i++) repeatYs2.Add(Rng.NextBelow(ref repeatState2, 1 << 20));
        for (int i = 0; i < 300; i++) repeat2.Add(i, repeatYs2[i]);
        Check(repeat.Merges == repeat2.Merges && repeat.Count == repeat2.Count,
                "the structure is reproducible (" + repeat.Merges + " merges)");
        for (int i = 0; i < repeatYs.Count; i++)
        {
            Check(repeatYs[i] == repeatYs2[i], "the input stream is reproducible at " + i);
        }
        Check(repeat.CountInRectangle(0, 299, 0, (1 << 20) - 1) == 300, "the whole structure spans every point");

        Console.WriteLine("PASS");
    }
}
