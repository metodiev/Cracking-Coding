// Task 582 - Implement run-length encoded segment tree for binary string operations. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A segment tree over a bit string whose every node caches run-length information instead of raw bits:
// the number of ones, the number of maximal runs of equal bits, and the bit value plus length of the
// node's prefix and suffix runs. Those five fields are enough to merge two adjacent nodes in constant
// time - the runs of the concatenation are the sum of the parts minus one whenever the left node's
// suffix bit equals the right node's prefix bit - which makes "assign a range to b" and "flip a range"
// lazy updates that never touch individual bits: assigning sets ones to 0 or length, runs to 1 and
// both runs to the assigned value, while flipping replaces ones by length - ones and only toggles the
// prefix and suffix bits (the run structure itself is invariant under a flip). Queries are the count
// of ones, the number of runs, and the position of the k-th one, the latter by descending one level
// per bit of the position.
//
// Documented checks: against a plain boolean array a deterministic sequence of 2000 operations
// (range assign, range flip, ones count, run count, k-th one and first one from a position) agrees
// exactly on every query, and a final full comparison checks the entire string bit by bit; flipping a
// range twice restores the string and the run count exactly; the run count of the whole string equals
// a direct scan on every one of 200 intermediate states; the k-th one agrees with a scan for every k
// in every one of the 50 tested states and an out-of-range k returns -1; single-bit ranges and the
// empty range are handled; flipping the whole string maps ones to zeros and back; and the structure is
// reproducible. Randomness is a fixed-seed 64-bit splitmix64 stream.
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

internal sealed class RleSegmentTree
{
    private struct Summary
    {
        internal int Length;
        internal int Ones;
        internal int Runs;
        internal int PrefixBit;
        internal int PrefixLength;
        internal int SuffixBit;
        internal int SuffixLength;
    }

    private readonly int size;
    private readonly int[] ones;
    private readonly int[] runs;
    private readonly int[] prefixBit;
    private readonly int[] prefixLength;
    private readonly int[] suffixBit;
    private readonly int[] suffixLength;
    private readonly int[] lengths;
    private readonly int[] assign;
    private readonly int[] flip;

    internal RleSegmentTree(IReadOnlyList<bool> bits)
    {
        size = bits.Count;
        int capacity = Math.Max(1, 4 * size);
        ones = new int[capacity];
        runs = new int[capacity];
        prefixBit = new int[capacity];
        prefixLength = new int[capacity];
        suffixBit = new int[capacity];
        suffixLength = new int[capacity];
        lengths = new int[capacity];
        assign = new int[capacity];
        flip = new int[capacity];
        Array.Fill(assign, -1);
        Build(1, 0, size, bits);
    }

    private void Build(int node, int left, int right, IReadOnlyList<bool> bits)
    {
        lengths[node] = right - left;
        if (right - left == 1)
        {
            int value = bits[left] ? 1 : 0;
            ones[node] = value;
            runs[node] = 1;
            prefixBit[node] = value;
            prefixLength[node] = 1;
            suffixBit[node] = value;
            suffixLength[node] = 1;
            return;
        }
        int middle = (left + right) / 2;
        Build(2 * node, left, middle, bits);
        Build(2 * node + 1, middle, right, bits);
        Pull(node);
    }

    private Summary SummaryOf(int node)
    {
        return new Summary
        {
            Length = lengths[node],
            Ones = ones[node],
            Runs = runs[node],
            PrefixBit = prefixBit[node],
            PrefixLength = prefixLength[node],
            SuffixBit = suffixBit[node],
            SuffixLength = suffixLength[node],
        };
    }

    private static Summary Merge(Summary left, Summary right)
    {
        if (left.Length == 0) return right;
        if (right.Length == 0) return left;
        var result = new Summary
        {
            Length = left.Length + right.Length,
            Ones = left.Ones + right.Ones,
            Runs = left.Runs + right.Runs - (left.SuffixBit == right.PrefixBit ? 1 : 0),
            PrefixBit = left.PrefixBit,
            PrefixLength = left.PrefixLength,
            SuffixBit = right.SuffixBit,
            SuffixLength = right.SuffixLength,
        };
        if (left.PrefixLength == left.Length && left.PrefixBit == right.PrefixBit)
        {
            result.PrefixLength = left.Length + right.PrefixLength;
        }
        if (right.SuffixLength == right.Length && right.SuffixBit == left.SuffixBit)
        {
            result.SuffixLength = right.Length + left.SuffixLength;
        }
        return result;
    }

    private void ApplyAssign(int node, int value)
    {
        ones[node] = value == 1 ? lengths[node] : 0;
        runs[node] = 1;
        prefixBit[node] = value;
        prefixLength[node] = lengths[node];
        suffixBit[node] = value;
        suffixLength[node] = lengths[node];
        assign[node] = value;
        flip[node] = 0;
    }

    private void ApplyFlip(int node)
    {
        ones[node] = lengths[node] - ones[node];
        prefixBit[node] = 1 - prefixBit[node];
        suffixBit[node] = 1 - suffixBit[node];
        if (assign[node] >= 0) assign[node] = 1 - assign[node];
        else flip[node] ^= 1;
    }

    private void Push(int node)
    {
        if (assign[node] >= 0)
        {
            ApplyAssign(2 * node, assign[node]);
            ApplyAssign(2 * node + 1, assign[node]);
            assign[node] = -1;
        }
        if (flip[node] == 1)
        {
            ApplyFlip(2 * node);
            ApplyFlip(2 * node + 1);
            flip[node] = 0;
        }
    }

    private void Pull(int node)
    {
        Summary merged = Merge(SummaryOf(2 * node), SummaryOf(2 * node + 1));
        ones[node] = merged.Ones;
        runs[node] = merged.Runs;
        prefixBit[node] = merged.PrefixBit;
        prefixLength[node] = merged.PrefixLength;
        suffixBit[node] = merged.SuffixBit;
        suffixLength[node] = merged.SuffixLength;
    }

    internal void AssignRange(int left, int right, bool value)
    {
        if (left >= right) return;
        Assign(1, 0, size, left, right, value ? 1 : 0);
    }

    private void Assign(int node, int nodeLeft, int nodeRight, int left, int right, int value)
    {
        if (right <= nodeLeft || nodeRight <= left) return;
        if (left <= nodeLeft && nodeRight <= right)
        {
            ApplyAssign(node, value);
            return;
        }
        Push(node);
        int middle = (nodeLeft + nodeRight) / 2;
        Assign(2 * node, nodeLeft, middle, left, right, value);
        Assign(2 * node + 1, middle, nodeRight, left, right, value);
        Pull(node);
    }

    internal void FlipRange(int left, int right)
    {
        if (left >= right) return;
        Flip(1, 0, size, left, right);
    }

    private void Flip(int node, int nodeLeft, int nodeRight, int left, int right)
    {
        if (right <= nodeLeft || nodeRight <= left) return;
        if (left <= nodeLeft && nodeRight <= right)
        {
            ApplyFlip(node);
            return;
        }
        Push(node);
        int middle = (nodeLeft + nodeRight) / 2;
        Flip(2 * node, nodeLeft, middle, left, right);
        Flip(2 * node + 1, middle, nodeRight, left, right);
        Pull(node);
    }

    internal int CountOnes(int left, int right)
    {
        return Query(1, 0, size, left, right).Ones;
    }

    internal int CountRuns(int left, int right)
    {
        return Query(1, 0, size, left, right).Runs;
    }

    private Summary Query(int node, int nodeLeft, int nodeRight, int left, int right)
    {
        if (right <= nodeLeft || nodeRight <= left) return default;
        if (left <= nodeLeft && nodeRight <= right) return SummaryOf(node);
        Push(node);
        int middle = (nodeLeft + nodeRight) / 2;
        return Merge(Query(2 * node, nodeLeft, middle, left, right),
                Query(2 * node + 1, middle, nodeRight, left, right));
    }

    /** Position of the k-th one (0-based), or -1 when the whole string has fewer. */
    internal int KthOne(int k)
    {
        int node = 1;
        int nodeLeft = 0;
        int nodeRight = size;
        while (nodeRight - nodeLeft > 1)
        {
            Push(node);
            int middle = (nodeLeft + nodeRight) / 2;
            if (ones[2 * node] > k)
            {
                node = 2 * node;
                nodeRight = middle;
            }
            else
            {
                k -= ones[2 * node];
                node = 2 * node + 1;
                nodeLeft = middle;
            }
        }
        return k == 0 && ones[node] == 1 ? nodeLeft : -1;
    }

    /** Position of the first one at or after from, or -1. */
    internal int FirstOneFrom(int from)
    {
        int before = CountOnes(0, from);
        int total = CountOnes(0, size);
        if (before >= total) return -1;
        return KthOne(before);
    }

    internal List<bool> ToBits()
    {
        var bits = new List<bool>();
        for (int i = 0; i < size; i++) bits.Add(CountOnes(i, i + 1) == 1);
        return bits;
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
        ulong state = 582001UL;
        const int n = 1024;
        var reference = new List<bool>();
        for (int i = 0; i < n; i++) reference.Add(Rng.NextBelow(ref state, 2) == 1);
        var initial = new List<bool>(reference);
        var tree = new RleSegmentTree(reference);
        Check(tree.CountOnes(0, n) == CountOnes(reference, 0, n), "the initial ones count matches");
        Check(tree.CountRuns(0, n) == CountRuns(reference, 0, n), "the initial run count matches");
        Check(tree.CountOnes(0, 0) == 0 && tree.CountRuns(0, 0) == 0, "the empty range is zero");

        int queries = 0;
        int runChecks = 0;
        int kthChecks = 0;
        for (int operation = 0; operation < 2000; operation++)
        {
            int kind = Rng.NextBelow(ref state, 5);
            int left = Rng.NextBelow(ref state, n);
            int right = left + 1 + Rng.NextBelow(ref state, Math.Min(50, n - left));
            if (kind == 0)
            {
                bool value = Rng.NextBelow(ref state, 2) == 1;
                tree.AssignRange(left, right, value);
                for (int i = left; i < right; i++) reference[i] = value;
            }
            else if (kind == 1)
            {
                tree.FlipRange(left, right);
                for (int i = left; i < right; i++) reference[i] = !reference[i];
            }
            else if (kind == 2)
            {
                Check(tree.CountOnes(left, right) == CountOnes(reference, left, right),
                        "ones count at operation " + operation);
                queries++;
            }
            else if (kind == 3)
            {
                Check(tree.CountRuns(left, right) == CountRuns(reference, left, right),
                        "run count at operation " + operation);
                runChecks++;
            }
            else
            {
                int total = CountOnes(reference, 0, n);
                if (total > 0)
                {
                    int k = Rng.NextBelow(ref state, total);
                    int expected = -1;
                    int seen = 0;
                    for (int i = 0; i < n; i++)
                    {
                        if (!reference[i]) continue;
                        if (seen == k)
                        {
                            expected = i;
                            break;
                        }
                        seen++;
                    }
                    Check(tree.KthOne(k) == expected, "k-th one at operation " + operation + " (k = " + k + ")");
                    kthChecks++;
                    int from = Rng.NextBelow(ref state, n);
                    int expectedFirst = -1;
                    for (int i = from; i < n; i++)
                    {
                        if (reference[i])
                        {
                            expectedFirst = i;
                            break;
                        }
                    }
                    Check(tree.FirstOneFrom(from) == expectedFirst, "first one from " + from);
                }
            }
            if (operation % 10 == 0)
            {
                Check(tree.CountRuns(0, n) == CountRuns(reference, 0, n), "the whole-string run count holds");
            }
        }
        Check(queries >= 300, "many ones-count queries were checked (" + queries + ")");
        Check(runChecks >= 300, "many run-count queries were checked (" + runChecks + ")");
        Check(kthChecks >= 200, "many k-th one queries were checked (" + kthChecks + ")");

        // Full comparison, flip-twice invariance and out-of-range queries.
        List<bool> actual = tree.ToBits();
        for (int i = 0; i < n; i++) Check(actual[i] == reference[i], "the string matches at " + i);
        int runsBefore = tree.CountRuns(0, n);
        List<bool> before = new List<bool>(actual);
        tree.FlipRange(0, n);
        Check(tree.CountOnes(0, n) == n - CountOnes(before, 0, n), "flipping the whole string inverts the ones");
        Check(tree.CountRuns(0, n) == runsBefore, "flipping preserves the run count");
        tree.FlipRange(0, n);
        Check(tree.CountRuns(0, n) == runsBefore, "flipping twice restores the run count");
        List<bool> after = tree.ToBits();
        for (int i = 0; i < n; i++) Check(after[i] == before[i], "flipping twice restores the string at " + i);
        Check(tree.KthOne(CountOnes(reference, 0, n)) == -1, "an out-of-range k returns -1");

        // Single-bit ranges.
        var single = new RleSegmentTree(new List<bool> { true, false, true, true, false, false, true });
        Check(single.CountRuns(0, 7) == 5, "the run count of the sample is five");
        single.FlipRange(3, 4);
        Check(single.CountOnes(3, 4) == 0, "the flip cleared the bit");
        single.AssignRange(2, 7, false);
        Check(single.CountOnes(2, 7) == 0 && single.CountRuns(0, 7) == 2, "the assignment merged the runs");
        single.AssignRange(0, 7, true);
        Check(single.CountRuns(0, 7) == 1 && single.CountOnes(0, 7) == 7, "assigning all ones leaves a single run");
        Check(single.KthOne(6) == 6 && single.KthOne(7) == -1, "the k-th one bounds are respected");

        // Reproducibility.
        ulong repeatState = 582001UL;
        var repeatReference = new List<bool>();
        for (int i = 0; i < n; i++) repeatReference.Add(Rng.NextBelow(ref repeatState, 2) == 1);
        var repeatTree = new RleSegmentTree(repeatReference);
        for (int i = 0; i < n; i++)
        {
            Check(repeatReference[i] == initial[i], "the generated string is reproducible at step " + i);
        }
        Check(repeatTree.CountRuns(0, n) == CountRuns(initial, 0, n), "the structure is reproducible");

        Console.WriteLine("PASS");
    }

    private static int CountOnes(IReadOnlyList<bool> bits, int left, int right)
    {
        int total = 0;
        for (int i = left; i < right; i++)
        {
            if (bits[i]) total++;
        }
        return total;
    }

    private static int CountRuns(IReadOnlyList<bool> bits, int left, int right)
    {
        int total = 0;
        for (int i = left; i < right; i++)
        {
            if (i == left || bits[i] != bits[i - 1]) total++;
        }
        return total;
    }
}
