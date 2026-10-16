// Task 579 - Implement wavelet tree for strings (rank/select). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A wavelet tree over a string: the alphabet is mapped to 0..sigma-1, each internal node stores a bit
// vector telling which elements go to the left child (with prefix sums of the ones), and the tree
// splits the sequence by one bit per level until only a single character remains, which becomes a
// leaf holding the characters that reached it. That gives the four classic operations: Access(i) by
// descending one level per bit; Rank(c, i) - the number of occurrences of c in s[0..i) - by counting
// the elements that reach c's leaf while mapping the index range down; Select(c, j) - the position of
// the j-th occurrence of c - by descending to c's leaf and then climbing back up with select on the
// bit vectors; and the range queries RangeCount(c, l, r) and KthSmallest(l, r, k), the latter by
// counting how many of the range's elements took the zero branch at each level.
//
// Documented checks: on a 64-character string over a four-letter alphabet every RangeCount(c, l, r)
// over all 2080 index ranges agrees with a direct scan, every Rank(c, i) over all i and all c matches
// a brute-force prefix count, every KthSmallest(l, r, k) equals the k-th element of the sorted
// substring, and Select is exactly the inverse of Rank - Rank(c, Select(c, j)) = j + 1 and
// Select(c, Rank(c, i)) = i whenever s[i] = c - for all characters and all occurrences, while an
// out-of-range j returns -1; Access reproduces the string character by character; the depth of the
// tree is the alphabet's bit width and the leaf sizes sum to the string length; a two-letter string
// (the minimum alphabet) and a single-character string are handled; and the whole structure is
// reproducible. The string is generated from a fixed-seed 64-bit splitmix64 stream.
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

internal sealed class StringWaveletTree
{
    private sealed class Node
    {
        internal int Bit;
        internal int[] Bits = Array.Empty<int>();
        internal int[] PrefixOnes = Array.Empty<int>();
        internal Node? Left;
        internal Node? Right;
    }

    private readonly Node? root;
    internal readonly int Length;
    internal readonly int BitWidth;

    internal StringWaveletTree(IReadOnlyList<int> text, int alphabet)
    {
        Length = text.Count;
        BitWidth = 1;
        while ((1 << BitWidth) < Math.Max(2, alphabet)) BitWidth++;
        root = Build(text, BitWidth - 1);
    }

    private static Node? Build(IReadOnlyList<int> values, int bit)
    {
        if (values.Count == 0) return null;
        var node = new Node { Bit = bit };
        if (bit < 0)
        {
            node.Bits = new int[values.Count];
            node.PrefixOnes = new int[values.Count + 1];
            return node;
        }
        node.Bits = new int[values.Count];
        node.PrefixOnes = new int[values.Count + 1];
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

    internal int Access(int index)
    {
        if (index < 0 || index >= Length) return -1;
        Node? node = root;
        int value = 0;
        while (node != null && node.Bit >= 0)
        {
            int position = index;
            if (node.Bits[position] == 1)
            {
                value |= 1 << node.Bit;
                index = node.PrefixOnes[position + 1] - 1;
                node = node.Right;
            }
            else
            {
                index = position - node.PrefixOnes[position];
                node = node.Left;
            }
        }
        return node == null ? -1 : value;
    }

    /** Number of occurrences of value in the index range [left, right). */
    internal int RangeCount(int value, int left, int right)
    {
        return Rank(value, right) - Rank(value, left);
    }

    /** Number of occurrences of value in s[0..index). */
    internal int Rank(int value, int index)
    {
        if (index <= 0) return 0;
        if (index > Length) index = Length;
        Node? node = root;
        int low = 0;
        int high = index;
        while (node != null && node.Bit >= 0)
        {
            int onesLow = node.PrefixOnes[low];
            int onesHigh = node.PrefixOnes[high];
            if ((value & (1 << node.Bit)) == 0)
            {
                low -= onesLow;
                high -= onesHigh;
                node = node.Left;
            }
            else
            {
                low = onesLow;
                high = onesHigh;
                node = node.Right;
            }
        }
        return high - low;
    }

    private static int SelectOnes(Node node, int j)
    {
        int seen = 0;
        for (int i = 0; i < node.Bits.Length; i++)
        {
            if (node.Bits[i] != 1) continue;
            if (seen == j) return i;
            seen++;
        }
        return -1;
    }

    private static int SelectZeros(Node node, int j)
    {
        int seen = 0;
        for (int i = 0; i < node.Bits.Length; i++)
        {
            if (node.Bits[i] != 0) continue;
            if (seen == j) return i;
            seen++;
        }
        return -1;
    }

    /** Position of the j-th occurrence (0-based) of value, or -1 when there is none. */
    internal int Select(int value, int j)
    {
        if (j < 0) return -1;
        var path = new List<(Node Node, bool Right)>();
        Node? node = root;
        while (node != null && node.Bit >= 0)
        {
            bool right = ((value >> node.Bit) & 1) == 1;
            path.Add((node, right));
            node = right ? node.Right : node.Left;
        }
        if (node == null || j >= node.Bits.Length) return -1;
        int index = j;
        for (int level = path.Count - 1; level >= 0; level--)
        {
            (Node parent, bool right) = path[level];
            index = right ? SelectOnes(parent, index) : SelectZeros(parent, index);
            if (index < 0) return -1;
        }
        return index;
    }

    /** The k-th smallest value (0-based) in [left, right). */
    internal int KthSmallest(int left, int right, int k)
    {
        if (left < 0 || right > Length || left >= right || k < 0 || k >= right - left) return -1;
        Node? node = root;
        int value = 0;
        while (node != null && node.Bit >= 0)
        {
            int onesLow = node.PrefixOnes[left];
            int onesHigh = node.PrefixOnes[right];
            int zeros = (right - left) - (onesHigh - onesLow);
            if (k < zeros)
            {
                left -= onesLow;
                right -= onesHigh;
                node = node.Left;
            }
            else
            {
                k -= zeros;
                value |= 1 << node.Bit;
                left = onesLow;
                right = onesHigh;
                node = node.Right;
            }
        }
        return value;
    }

    internal int LeafSizeSum()
    {
        return LeafSizeSum(root);
    }

    private static int LeafSizeSum(Node? node)
    {
        if (node == null) return 0;
        if (node.Bit < 0) return node.Bits.Length;
        return LeafSizeSum(node.Left) + LeafSizeSum(node.Right);
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
        ulong state = 579001UL;
        var text = new List<int>();
        for (int i = 0; i < 64; i++) text.Add(Rng.NextBelow(ref state, 4));
        var tree = new StringWaveletTree(text, 4);
        Check(tree.Length == 64, "the tree holds 64 characters");
        Check(tree.BitWidth == 2, "a four-letter alphabet needs two bits per level");
        Check(tree.LeafSizeSum() == 64, "the leaf sizes sum to the string length");
        for (int i = 0; i < 64; i++) Check(tree.Access(i) == text[i], "Access reproduces the string at " + i);
        Check(tree.Access(-1) == -1 && tree.Access(64) == -1, "Access outside the string returns -1");

        int ranges = 0;
        for (int left = 0; left < 64; left++)
        {
            for (int right = left + 1; right <= 64; right++)
            {
                for (int value = 0; value < 4; value++)
                {
                    int expected = 0;
                    for (int i = left; i < right; i++)
                    {
                        if (text[i] == value) expected++;
                    }
                    Check(tree.RangeCount(value, left, right) == expected, "RangeCount(" + value + ", " + left + ", "
                            + right + ") matches a direct scan");
                    ranges++;
                }
                for (int k = 0; k < right - left; k++)
                {
                    var sorted = new List<int>();
                    for (int i = left; i < right; i++) sorted.Add(text[i]);
                    sorted.Sort();
                    Check(tree.KthSmallest(left, right, k) == sorted[k], "KthSmallest(" + left + ", " + right + ", "
                            + k + ") matches the sorted substring");
                }
            }
        }
        Check(ranges == 2080 * 4, "every range and character was checked (" + ranges + ")");
        Check(tree.KthSmallest(0, 64, -1) == -1 && tree.KthSmallest(0, 64, 64) == -1,
                "KthSmallest outside the range returns -1");
        Check(tree.KthSmallest(10, 10, 0) == -1, "an empty range has no k-th element");

        for (int value = 0; value < 4; value++)
        {
            int count = tree.Rank(value, 64);
            Check(tree.Rank(value, 0) == 0, "Rank at zero is zero for " + value);
            Check(tree.Select(value, count) == -1, "Select past the last occurrence returns -1 for " + value);
            for (int j = 0; j < count; j++)
            {
                int position = tree.Select(value, j);
                Check(position >= 0 && position < 64, "Select returns a valid position for " + value + ", " + j);
                Check(text[position] == value, "Select lands on the right character for " + value + ", " + j);
                Check(tree.Rank(value, position + 1) == j + 1, "Rank of Select is j + 1 for " + value + ", " + j);
            }
            for (int i = 1; i <= 64; i++)
            {
                if (i > 0 && text[i - 1] == value)
                {
                    int rank = tree.Rank(value, i);
                    Check(tree.Select(value, rank - 1) == i - 1, "Select inverts Rank at " + i);
                }
            }
        }
        Check(tree.Select(0, -1) == -1, "a negative select index returns -1");

        // A two-letter and a single-character string.
        var binary = new List<int>();
        for (int i = 0; i < 33; i++) binary.Add(i % 2);
        var binaryTree = new StringWaveletTree(binary, 2);
        Check(binaryTree.BitWidth == 1, "a two-letter alphabet needs one bit");
        Check(binaryTree.Rank(1, 33) == 16, "the alternating string has sixteen ones");
        Check(binaryTree.Select(1, 0) == 1 && binaryTree.Select(1, 15) == 31, "select works on the binary string");
        Check(binaryTree.KthSmallest(0, 33, 16) == 0 && binaryTree.KthSmallest(0, 33, 17) == 1,
                "the seventeenth smallest character is a zero and the eighteenth is a one");
        var single = new StringWaveletTree(new List<int> { 3, 3, 3 }, 4);
        Check(single.Rank(3, 3) == 3 && single.Select(3, 2) == 2, "a constant string is handled");
        Check(single.RangeCount(1, 0, 3) == 0, "an absent character counts zero");

        // Reproducibility.
        ulong repeatState = 579001UL;
        var repeatText = new List<int>();
        for (int i = 0; i < 64; i++) repeatText.Add(Rng.NextBelow(ref repeatState, 4));
        var repeatTree = new StringWaveletTree(repeatText, 4);
        for (int i = 0; i < 64; i++)
        {
            Check(repeatText[i] == text[i], "the generated string is reproducible at " + i);
            Check(repeatTree.Access(i) == tree.Access(i), "the tree is reproducible at " + i);
        }

        Console.WriteLine("PASS");
    }
}
