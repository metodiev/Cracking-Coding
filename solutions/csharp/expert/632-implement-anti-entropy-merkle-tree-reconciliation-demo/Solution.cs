// Task 632 - Implement anti-entropy Merkle tree reconciliation demo. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// Anti-entropy between two replicas compares a Merkle tree instead of the whole data set. The key
// space is split into fixed buckets (leaves); an internal node's hash is the hash of its children's
// hashes, so any change to a leaf changes every hash on the path to the root. Two replicas first
// exchange roots: equal roots mean identical content and the exchange stops after ONE comparison.
// Otherwise both descend together, and because a subtree whose hashes agree cannot contain a
// difference, only the paths that lead to differing leaves are visited - O(d log n) comparisons for
// d differences, instead of O(n).
//
// What is simplified (documented per the authoring contract): a single process, no threads or
// sockets and no incremental rehashing (a replica rebuilds its tree after a change, which is what
// an anti-entropy round would do anyway); keys are small integers and values are ints so the bucket
// layout is obvious. FNV-1a is used as the hash; the security argument (that a hash collision is
// what would break reconciliation) is not claimed, only the data-structure property is verified.
// Verified invariants: identical replicas compare in one hash, the differing buckets found are
// exactly the brute-force set difference, a written value changes every ancestor hash up to the
// root, and applying the differences makes both roots equal again.
using System;
using System.Collections.Generic;
using System.Text;

#nullable enable

internal sealed class MerkleTree
{
    public readonly int Size;          // number of leaves, a power of two
    public readonly long[] Nodes;      // heap layout: leaves live in [Size, 2*Size)
    public readonly int[] Buckets;     // a copy of the hashed content, for reconciliation

    public MerkleTree(int[] buckets, int size)
    {
        Size = size;
        Nodes = new long[2 * size];
        Buckets = new int[size];
        Array.Copy(buckets, Buckets, size);
        for (int i = 0; i < size; i++)
        {
            Nodes[size + i] = HashLeaf(i, buckets[i]);
        }
        for (int i = size - 1; i >= 1; i--)
        {
            Nodes[i] = Combine(Nodes[2 * i], Nodes[2 * i + 1]);
        }
    }

    public long Root => Nodes[1];
    public int LeafCount => Size;

    /// <summary>FNV-1a over the bucket index, the presence flag and the value.</summary>
    public static long HashLeaf(int key, int value)
    {
        long hash = 1469598103934665603L;
        hash = MixInt(hash, key);
        hash = MixInt(hash, value == -1 ? 0 : 1);
        hash = MixInt(hash, value == -1 ? 0 : value);
        return hash;
    }

    public static long Combine(long left, long right)
    {
        long hash = 1469598103934665603L;
        for (int i = 0; i < 8; i++) hash = Mix(hash, (int)((left >> (i * 8)) & 0xFF));
        for (int i = 0; i < 8; i++) hash = Mix(hash, (int)((right >> (i * 8)) & 0xFF));
        return hash;
    }

    private static long Mix(long hash, int value)
    {
        hash ^= value & 0xFF;
        hash *= 1099511628211L;
        return hash;
    }

    /// <summary>Mix all four bytes: folding to one byte would make 169 and 1193 collide.</summary>
    private static long MixInt(long hash, int value)
    {
        for (int i = 0; i < 4; i++) hash = Mix(hash, (value >> (i * 8)) & 0xFF);
        return hash;
    }

    /// <summary>Build a tree over a key space padded up to a power of two.</summary>
    public static MerkleTree Build(Dictionary<int, int> content, int keyCount)
    {
        int size = 1;
        while (size < keyCount) size <<= 1;
        var buckets = new int[size];
        for (int key = 0; key < size; key++) buckets[key] = -1;
        foreach (KeyValuePair<int, int> kv in content) buckets[kv.Key] = kv.Value;
        return new MerkleTree(buckets, size);
    }

    /// <summary>Compare two trees top-down; returns the differing leaf keys and the comparison count.</summary>
    public static List<int> DifferingKeys(MerkleTree a, MerkleTree b, out int comparisons)
    {
        comparisons = 0;
        var differing = new List<int>();
        if (a.Size != b.Size) throw new Exception("the two trees must cover the same key space");
        comparisons++;                                    // the root exchange
        if (a.Root == b.Root) return differing;
        var stack = new Stack<int>();
        stack.Push(1);
        while (stack.Count > 0)
        {
            int index = stack.Pop();
            comparisons++;
            if (a.Nodes[index] == b.Nodes[index]) continue;             // identical subtree: prune
            if (index >= a.Size)
            {
                differing.Add(index - a.Size);
                continue;
            }
            stack.Push(2 * index);
            stack.Push(2 * index + 1);
        }
        differing.Sort();
        return differing;
    }

    public static Dictionary<int, int> Content(Dictionary<int, int> source) => new Dictionary<int, int>(source);

    public static string Describe(Dictionary<int, int> content)
    {
        var keys = new List<int>(content.Keys);
        keys.Sort();
        var sb = new StringBuilder();
        foreach (int key in keys) sb.Append(key).Append('=').Append(content[key]).Append(' ');
        return sb.ToString().TrimEnd();
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int NextSeed(int seed) => (seed * 1103515245 + 12345) & 0x7FFFFFFF;

    private static Dictionary<int, int> MakeContent(int keyCount, int seed, int mutationKey, int mutationValue)
    {
        var content = new Dictionary<int, int>();
        for (int key = 0; key < keyCount; key++) content[key] = key * 3 + 1;
        if (mutationKey >= 0)
        {
            if (mutationValue < 0) content.Remove(mutationKey);
            else content[mutationKey] = mutationValue;
        }
        return content;
    }

    private static void IdenticalReplicasCompareInOneHash()
    {
        Dictionary<int, int> content = MakeContent(64, 1, -1, 0);
        MerkleTree a = MerkleTree.Build(content, 64);
        MerkleTree b = MerkleTree.Build(MerkleTree.Content(content), 64);
        Check(a.Root == b.Root, "identical content has identical roots");
        List<int> differing = MerkleTree.DifferingKeys(a, b, out int comparisons);
        Check(differing.Count == 0, "no differences");
        Check(comparisons == 1, "a single root comparison settles it, got " + comparisons);
        Check(a.Nodes[1] == MerkleTree.Combine(a.Nodes[2], a.Nodes[3]), "the root hashes its children");
        Check(a.Root != MerkleTree.Build(MakeContent(64, 1, 5, 999), 64).Root, "any change moves the root");
    }

    private static void SingleDifferenceIsFoundLogarithmically()
    {
        int keyCount = 1024;
        Dictionary<int, int> content = MakeContent(keyCount, 2, -1, 0);
        MerkleTree left = MerkleTree.Build(content, keyCount);
        MerkleTree right = MerkleTree.Build(MakeContent(keyCount, 2, 777, 4242), keyCount);
        Check(left.Root != right.Root, "the roots differ");
        List<int> differing = MerkleTree.DifferingKeys(left, right, out int comparisons);
        Check(differing.Count == 1 && differing[0] == 777, "exactly the changed key is reported");
        Check(comparisons <= 40, "one difference out of 1024 costs " + comparisons + " comparisons (logarithmic)");
        Check(comparisons < keyCount / 10, "far cheaper than scanning every bucket");
        Console.WriteLine("merkle: 1 difference in 1024 buckets = " + comparisons + " hash comparisons");
    }

    private static void DifferencesMatchTheBruteForceSetDifference()
    {
        int keyCount = 256;
        int seed = 99;
        for (int trial = 0; trial < 12; trial++)
        {
            seed = NextSeed(seed);
            var leftContent = MakeContent(keyCount, seed, -1, 0);
            var rightContent = MakeContent(keyCount, seed, -1, 0);
            int changes = 1 + seed % 6;
            for (int i = 0; i < changes; i++)
            {
                seed = NextSeed(seed);
                int key = seed % keyCount;
                seed = NextSeed(seed);
                if (seed % 5 == 0) rightContent.Remove(key);
                else rightContent[key] = 1000 + seed % 500;
            }
            // The expected set is derived from the final contents, so a mutation that happens to
            // restore the original value does not count as a difference.
            var expected = new List<int>();
            for (int key = 0; key < keyCount; key++)
            {
                bool inLeft = leftContent.TryGetValue(key, out int leftValue);
                bool inRight = rightContent.TryGetValue(key, out int rightValue);
                if (inLeft != inRight || (inLeft && leftValue != rightValue)) expected.Add(key);
            }
            expected.Sort();
            MerkleTree left = MerkleTree.Build(leftContent, keyCount);
            MerkleTree right = MerkleTree.Build(rightContent, keyCount);
            List<int> differing = MerkleTree.DifferingKeys(left, right, out int comparisons);
            Check(string.Join(",", differing) == string.Join(",", expected),
                "trial " + trial + ": found [" + string.Join(",", differing) + "] expected [" + string.Join(",", expected) + "]");
            if (expected.Count > 0)
            {
                Check(comparisons < keyCount, "trial " + trial + ": " + comparisons + " comparisons beat a full scan");
            }
            else
            {
                Check(comparisons == 1, "an unchanged pair costs one comparison");
            }
        }
    }

    private static void ReconciliationMakesTheTreesEqual()
    {
        int keyCount = 64;
        Dictionary<int, int> donor = MakeContent(keyCount, 7, -1, 0);
        donor[3] = 555;
        donor.Remove(9);
        donor[60] = 61;
        Dictionary<int, int> receiver = MakeContent(keyCount, 7, -1, 0);
        receiver[11] = 999;
        receiver.Remove(40);

        MerkleTree donorTree = MerkleTree.Build(donor, keyCount);
        MerkleTree receiverTree = MerkleTree.Build(receiver, keyCount);
        List<int> differing = MerkleTree.DifferingKeys(donorTree, receiverTree, out int comparisons);
        Check(differing.Count > 0, "the replicas differ");
        var expected = new List<int>();
        for (int key = 0; key < keyCount; key++)
        {
            bool inDonor = donor.TryGetValue(key, out int dv);
            bool inReceiver = receiver.TryGetValue(key, out int rv);
            if (inDonor != inReceiver || (inDonor && dv != rv)) expected.Add(key);
        }
        Check(string.Join(",", differing) == string.Join(",", expected), "the diff set is exactly the difference");

        // Apply the differences the way an anti-entropy round would.
        foreach (int key in differing)
        {
            if (donor.TryGetValue(key, out int value)) receiver[key] = value;
            else receiver.Remove(key);
        }
        MerkleTree repaired = MerkleTree.Build(receiver, keyCount);
        Check(repaired.Root == donorTree.Root, "the repaired replica matches the donor");
        Check(MerkleTree.Describe(receiver) == MerkleTree.Describe(donor), "and its content is identical");
        Check(MerkleTree.DifferingKeys(donorTree, repaired, out int after).Count == 0 && after == 1,
            "a second round costs one comparison");
        Console.WriteLine("merkle: reconciled " + differing.Count + " of " + keyCount + " buckets in "
            + comparisons + " comparisons");
    }

    private static void RootChangesPropagateToTheTop()
    {
        int keyCount = 32;
        Dictionary<int, int> content = MakeContent(keyCount, 3, -1, 0);
        MerkleTree before = MerkleTree.Build(content, keyCount);
        var changed = new Dictionary<int, int>(content) { [31] = 12345 };
        MerkleTree after = MerkleTree.Build(changed, keyCount);
        Check(after.Root != before.Root, "the root changed");
        int changedLeaves = 0;
        for (int i = 0; i < keyCount; i++)
        {
            if (before.Nodes[keyCount + i] != after.Nodes[keyCount + i]) changedLeaves++;
        }
        Check(changedLeaves == 1, "exactly one leaf hash changed");
        int unchangedInternal = 0;
        for (int i = 1; i < keyCount; i++)
        {
            if (before.Nodes[i] == after.Nodes[i]) unchangedInternal++;
        }
        Check(unchangedInternal == 26, "only the path ancestors changed, got " + unchangedInternal + " untouched");
        Check(before.Nodes[1] != after.Nodes[1] && before.Nodes[2] == after.Nodes[2],
            "the untouched half of the tree keeps its hash");

        // Boundary cases: an empty replica and a one-bucket key space.
        MerkleTree empty = MerkleTree.Build(new Dictionary<int, int>(), 1);
        MerkleTree one = MerkleTree.Build(new Dictionary<int, int> { [0] = 7 }, 1);
        Check(empty.Root != one.Root, "an empty replica differs from a filled one");
        Check(string.Join(",", MerkleTree.DifferingKeys(empty, one, out int small)) == "0" && small == 2,
            "the single bucket is reported as differing");
        Check(MerkleTree.DifferingKeys(empty, empty, out int same).Count == 0 && same == 1, "a replica matches itself");
        Console.WriteLine("merkle: mutation touched 1 leaf, left " + unchangedInternal + " internal nodes alone");
    }

    public static void Main()
    {
        IdenticalReplicasCompareInOneHash();
        SingleDifferenceIsFoundLogarithmically();
        DifferencesMatchTheBruteForceSetDifference();
        ReconciliationMakesTheTreesEqual();
        RootChangesPropagateToTheTop();
        Console.WriteLine("PASS");
    }
}
