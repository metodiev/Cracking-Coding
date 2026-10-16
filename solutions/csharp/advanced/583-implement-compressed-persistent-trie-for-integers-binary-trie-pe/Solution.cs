// Task 583 - Implement compressed persistent trie for integers (binary trie persistent). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A persistent Patricia trie over 32-bit integers. Nodes come in two shapes: a leaf holding one value,
// and an internal node holding the single bit position at which the values of its subtree split, so
// unary chains of the plain binary trie are collapsed - the "compressed" part. Inserting into version
// v creates version v + 1 by path copying: every node on the search path is duplicated with the new
// child, every other node is shared with the previous version, so a single insert allocates at most
// one node per level of the (compressed) trie and all older versions stay intact. Because the trie
// keeps the highest differing bit at each internal node, the classic maximum-XOR query is a greedy
// descent that always takes the branch whose bit differs from the query's bit.
//
// Documented checks: for every one of the 300 versions built by 300 insertions, membership matches the
// set of values inserted up to that version (a brute-force hash set is compared over the whole value
// pool at every second version), a value inserted later is absent from the earlier versions, and the
// empty version contains nothing; version 10 is still queryable, with exactly its ten values, after
// version 300 has been built, which is the persistence property; the number of nodes allocated per
// insert never exceeds one per trie level plus a leaf (bounded by 34, measured 13) and the total
// respects the same bound; the compressed trie is checked to use far fewer nodes than an equivalent
// uncompressed binary trie on a prefix-heavy data set (199 nodes against 1811 for values that differ
// only in bits 16..22). The cumulative allocation is honestly larger than a single plain trie - 2655
// nodes were created while building 302 versions - which is the expected price of path copying, so
// the compression comparison is made per version; the
// maximum-XOR query matches brute force over every version for 50 random queries, and the values are
// random 32-bit integers from a fixed-seed 64-bit splitmix64 stream, so no overflow or sign luck is
// involved.
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
}

internal sealed class PersistentTrie
{
    internal sealed class Node
    {
        internal int Bit;      // -1 for a leaf
        internal int Value;    // only meaningful for a leaf
        internal Node? Zero;
        internal Node? One;
        internal int Count;
    }

    private readonly List<Node?> roots = new List<Node?>();
    internal int NodesCreated;

    internal PersistentTrie()
    {
        roots.Add(null);
    }

    internal int VersionCount => roots.Count;

    internal int Count(int version)
    {
        Node? root = roots[version];
        return root == null ? 0 : root.Count;
    }

    private Node Leaf(int value)
    {
        NodesCreated++;
        return new Node { Bit = -1, Value = value, Count = 1 };
    }

    private Node Branch(int first, int second, int bit)
    {
        NodesCreated++;
        var node = new Node { Bit = bit, Count = 2 };
        if (((first >> bit) & 1) == 0)
        {
            node.Zero = Leaf(first);
            node.One = Leaf(second);
        }
        else
        {
            node.Zero = Leaf(second);
            node.One = Leaf(first);
        }
        return node;
    }

    private static int HighestDifferingBit(int a, int b)
    {
        int difference = a ^ b;
        int bit = 31;
        while (bit > 0 && (difference & (1 << bit)) == 0) bit--;
        return bit;
    }

    /** New version with the value added; duplicates leave the version unchanged. */
    internal int Insert(int version, int value)
    {
        Node? root = roots[version];
        if (Contains(version, value))
        {
            roots.Add(root);
            return roots.Count - 1;
        }
        roots.Add(InsertNode(root, value));
        return roots.Count - 1;
    }

    /** A representative value of a subtree: the leftmost leaf, which shares its top bits with all of them. */
    private static int Representative(Node node)
    {
        Node current = node;
        while (current.Bit >= 0) current = current.Zero!;
        return current.Value;
    }

    private Node InsertNode(Node? node, int value)
    {
        if (node == null) return Leaf(value);
        if (node.Bit < 0)
        {
            if (node.Value == value) return node;
            return Branch(node.Value, value, HighestDifferingBit(node.Value, value));
        }
        int divergent = HighestDifferingBit(Representative(node), value);
        if (divergent > node.Bit)
        {
            // The new value differs from the whole subtree above this node's bit, so the split has to be
            // placed at the higher bit and the existing subtree becomes one of the two children.
            NodesCreated++;
            var split = new Node { Bit = divergent, Count = node.Count + 1 };
            if (((value >> divergent) & 1) == 1)
            {
                split.Zero = node;
                split.One = Leaf(value);
            }
            else
            {
                split.Zero = Leaf(value);
                split.One = node;
            }
            return split;
        }
        bool goOne = ((value >> node.Bit) & 1) == 1;
        NodesCreated++;
        var created = new Node { Bit = node.Bit, Count = node.Count + 1 };
        if (goOne)
        {
            created.One = InsertNode(node.One, value);
            created.Zero = node.Zero;
        }
        else
        {
            created.Zero = InsertNode(node.Zero, value);
            created.One = node.One;
        }
        return created;
    }

    /** Number of distinct nodes reachable from a version's root: the size of that one trie. */
    internal int CountNodes(int version)
    {
        return CountNodes(roots[version]);
    }

    private static int CountNodes(Node? node)
    {
        if (node == null) return 0;
        return 1 + CountNodes(node.Zero) + CountNodes(node.One);
    }

    internal bool Contains(int version, int value)
    {
        Node? node = roots[version];
        while (node != null)
        {
            if (node.Bit < 0) return node.Value == value;
            node = ((value >> node.Bit) & 1) == 1 ? node.One : node.Zero;
        }
        return false;
    }

    /** Value in the version that maximises value XOR query; int.MinValue for an empty version. */
    internal int MaxXor(int version, int query)
    {
        Node? node = roots[version];
        if (node == null) return int.MinValue;
        while (node.Bit >= 0)
        {
            bool queryBit = ((query >> node.Bit) & 1) == 1;
            Node? preferred = queryBit ? node.Zero : node.One;
            if (preferred == null) preferred = queryBit ? node.One : node.Zero;
            node = preferred!;
        }
        return node.Value;
    }
}

/** Plain (uncompressed) binary trie over 32 bits, used only to measure what compression buys. */
internal sealed class UncompressedTrie
{
    private sealed class Node
    {
        internal Node? Zero;
        internal Node? One;
    }

    private readonly Node root = new Node();
    internal int Nodes = 1;

    internal void Add(int value)
    {
        Node current = root;
        for (int bit = 31; bit >= 0; bit--)
        {
            bool one = ((value >> bit) & 1) == 1;
            if (one)
            {
                if (current.One == null)
                {
                    current.One = new Node();
                    Nodes++;
                }
                current = current.One;
            }
            else
            {
                if (current.Zero == null)
                {
                    current.Zero = new Node();
                    Nodes++;
                }
                current = current.Zero;
            }
        }
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
        ulong state = 583001UL;
        var values = new List<int>();
        for (int i = 0; i < 300; i++) values.Add(unchecked((int)Rng.NextRaw(ref state)));

        var trie = new PersistentTrie();
        Check(trie.VersionCount == 1, "only the empty version exists at the start");
        Check(trie.Count(0) == 0 && !trie.Contains(0, values[0]), "the empty version contains nothing");
        var seen = new HashSet<int>();
        int worstNodes = 0;
        int previousNodes = trie.NodesCreated;
        for (int i = 0; i < values.Count; i++)
        {
            int version = trie.Insert(trie.VersionCount - 1, values[i]);
            seen.Add(values[i]);
            Check(version == i + 1, "the insert produced version " + (i + 1));
            Check(trie.Count(version) == seen.Count, "the version holds the right number of values at " + i
                    + " (" + trie.Count(version) + " vs " + seen.Count + ")");
            Check(trie.Contains(version, values[i]), "the inserted value is present in its own version");
            Check(trie.Contains(i + 1, values[0]), "the first value survives later versions");
            worstNodes = Math.Max(worstNodes, trie.NodesCreated - previousNodes);
            previousNodes = trie.NodesCreated;
        }
        Check(trie.VersionCount == 301, "three hundred and one versions exist");
        Check(worstNodes <= 34, "an insert allocates at most one node per trie level plus the leaf ("
                + worstNodes + ")");
        Check(trie.NodesCreated <= 34 * values.Count, "the total allocation respects the bound ("
                + trie.NodesCreated + ")");

        // Membership against brute force, and persistence of an old version.
        for (int version = 0; version <= values.Count; version += 2)
        {
            var expected = new HashSet<int>();
            for (int i = 0; i < version; i++) expected.Add(values[i]);
            foreach (int value in values)
            {
                Check(trie.Contains(version, value) == expected.Contains(value),
                        "membership at version " + version + " is right for " + value);
            }
            Check(trie.Count(version) == expected.Count, "the count at version " + version + " is right");
        }
        for (int i = 10; i < values.Count; i++)
        {
            Check(!trie.Contains(10, values[i]), "version 10 does not contain the later value at " + i);
        }
        Check(trie.Count(10) == 10, "version 10 still holds exactly ten values");
        Check(trie.Contains(300, values[299]), "the last version contains the last value");
        Check(!trie.Contains(300, values[299] ^ 1), "a neighbouring value is absent unless inserted");

        // Duplicate insertion must not create a new version with extra content.
        int lengthBefore = trie.VersionCount;
        int duplicateVersion = trie.Insert(lengthBefore - 1, values[0]);
        Check(trie.Count(duplicateVersion) == values.Count, "a duplicate insert changes nothing ("
                + trie.Count(duplicateVersion) + ")");

        // Maximum XOR against brute force.
        int xorChecks = 0;
        for (int trial = 0; trial < 50; trial++)
        {
            int query = unchecked((int)Rng.NextRaw(ref state));
            int version = (trial * 6) % (values.Count + 1);
            var expectedValues = new HashSet<int>();
            for (int i = 0; i < version; i++) expectedValues.Add(values[i]);
            if (expectedValues.Count == 0)
            {
                Check(trie.MaxXor(version, query) == int.MinValue, "an empty version has no XOR maximum");
                continue;
            }
            int bestValue = 0;
            uint bestXor = 0;
            foreach (int value in expectedValues)
            {
                // The maximum XOR is an unsigned maximum: the greedy descent maximises the highest
                // differing bit, which coincides with the unsigned order of the XOR values.
                uint xor = unchecked((uint)(value ^ query));
                if (xor > bestXor)
                {
                    bestXor = xor;
                    bestValue = value;
                }
            }
            int actual = trie.MaxXor(version, query);
            Check(unchecked((uint)(actual ^ query)) == bestXor, "the maximum XOR matches brute force at version "
                    + version + " (" + unchecked((uint)(actual ^ query)) + " vs " + bestXor + ")");
            Check(expectedValues.Contains(actual), "the returned value is in the version");
            xorChecks++;
        }
        Check(xorChecks >= 40, "many XOR queries were checked (" + xorChecks + ")");

        // Compression versus a plain trie on a prefix-heavy data set.
        // Values sharing a long prefix that differ only in bits 16..22: the plain trie must spend one
        // node per differing bit per value, while the Patricia trie only branches where the values
        // actually split, which is where compression pays off.
        var prefixHeavy = new List<int>();
        for (int i = 0; i < 100; i++) prefixHeavy.Add(unchecked((int)(0x10000000u + ((uint)i << 16))));
        var compressed = new PersistentTrie();
        var plain = new UncompressedTrie();
        foreach (int value in prefixHeavy)
        {
            compressed.Insert(compressed.VersionCount - 1, value);
            plain.Add(value);
        }
        int compressedNodes = compressed.CountNodes(compressed.VersionCount - 1);
        Check(compressedNodes <= plain.Nodes / 2, "one version of the compressed trie uses far fewer nodes than the "
                + "plain trie (" + compressedNodes + " vs " + plain.Nodes + ")");
        Check(compressed.Count(100) == 100, "the prefix-heavy version holds all 100 values");
        foreach (int value in prefixHeavy) Check(compressed.Contains(100, value), "the prefix-heavy value is present");
        Check(compressed.NodesCreated <= 34 * prefixHeavy.Count, "the persistent build stays inside the bound ("
                + compressed.NodesCreated + ")");

        // Reproducibility.
        ulong repeatState = 583001UL;
        var repeatValues = new List<int>();
        for (int i = 0; i < 300; i++) repeatValues.Add(unchecked((int)Rng.NextRaw(ref repeatState)));
        var repeatTrie = new PersistentTrie();
        foreach (int value in repeatValues) repeatTrie.Insert(repeatTrie.VersionCount - 1, value);
        Check(repeatTrie.NodesCreated == trie.NodesCreated, "the node allocation is reproducible ("
                + repeatTrie.NodesCreated + ")");
        for (int i = 0; i < values.Count; i++)
        {
            Check(repeatValues[i] == values[i], "the value stream is reproducible at " + i);
            Check(repeatTrie.Contains(i + 1, values[i]), "the repeated build has the same membership at " + i);
        }

        Console.WriteLine("PASS");
    }
}
