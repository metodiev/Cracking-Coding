// Task 474 - Implement Merkle tree for log integrity and proof generation. (Tier: Advanced, CTCI Beyond — Security & Privacy)
using System;
using System.Collections.Generic;

internal class MerkleTree
{
    public readonly List<string> Leaves;
    public readonly List<long[]> Levels;

    public MerkleTree(List<string> values)
    {
        Leaves = new List<string>(values);
        Levels = new List<long[]>();
        var level = new List<long>();
        foreach (string v in Leaves) level.Add(HashLeaf(v));
        Levels.Add(level.ToArray());
        while (level.Count > 1)
        {
            var next = new List<long>();
            for (int i = 0; i + 1 < level.Count; i += 2) next.Add(HashNode(level[i], level[i + 1]));
            if (level.Count % 2 == 1) next.Add(level[level.Count - 1]); // promote
            Levels.Add(next.ToArray());
            level = next;
        }
    }

    public long Root() => Levels[Levels.Count - 1][0];

    public List<long[]> Proof(int index)
    {
        var path = new List<long[]>();
        int idx = index;
        for (int li = 0; li < Levels.Count - 1; li++)
        {
            long[] level = Levels[li];
            if (idx % 2 == 1) path.Add(new long[] { level[idx - 1], 0 });        // sibling on the left
            else if (idx + 1 < level.Length) path.Add(new long[] { level[idx + 1], 1 });
            idx /= 2;
        }
        return path;
    }

    public static long Fnv1a(byte[] data)
    {
        long h = 2166136261L;
        foreach (byte b in data)
        {
            h = ((h ^ b) * 16777619L) & 0xFFFFFFFFL;
        }
        return h;
    }

    public static long HashLeaf(string value)
    {
        var data = new byte[value.Length + 1];
        for (int i = 0; i < value.Length; i++) data[i + 1] = (byte)value[i];
        return Fnv1a(data);
    }

    public static long HashNode(long left, long right)
    {
        var data = new byte[8];
        for (int i = 0; i < 4; i++)
        {
            data[i] = (byte)((left >> (8 * (3 - i))) & 0xFF);
            data[4 + i] = (byte)((right >> (8 * (3 - i))) & 0xFF);
        }
        return Fnv1a(data);
    }

    public static bool Verify(long root, string value, List<long[]> path)
    {
        long h = HashLeaf(value);
        foreach (long[] step in path)
        {
            h = step[1] == 0 ? HashNode(step[0], h) : HashNode(h, step[0]);
        }
        return h == root;
    }
}

public static class Solution
{
    public static void Main()
    {
        var leaves = new List<string>();
        for (int i = 0; i < 7; i++) leaves.Add("tx" + i);
        var tree = new MerkleTree(leaves);
        Check(tree.Root() == 818430076L, "root of the 7-record log");
        for (int i = 0; i < 7; i++)
        {
            Check(MerkleTree.Verify(tree.Root(), leaves[i], tree.Proof(i)), "proof must verify " + i);
        }
        Check(tree.Proof(0).Count == 3, "index 0 walks three levels");
        Check(tree.Proof(6).Count == 2, "promoted index 6 has no level-0 sibling");
        Check(!MerkleTree.Verify(tree.Root(), "tx3x", tree.Proof(3)), "tampered leaf must fail");
        Check(!MerkleTree.Verify(tree.Root() ^ 1L, leaves[3], tree.Proof(3)), "wrong root must fail");
        var bad = new List<long[]>(tree.Proof(3));
        bad[0] = new long[] { bad[0][0] ^ 1L, bad[0][1] };
        Check(!MerkleTree.Verify(tree.Root(), leaves[3], bad), "tampered sibling must fail");
        Check(new MerkleTree(leaves).Root() == tree.Root(), "fresh build matches");
        var rebuilt = new MerkleTree(leaves);
        for (int i = 0; i < 7; i++) Check(MerkleTree.Verify(rebuilt.Root(), leaves[i], tree.Proof(i)), "proofs agree");

        var leaves8 = new List<string>();
        for (int i = 0; i < 8; i++) leaves8.Add("tx" + i);
        var tree8 = new MerkleTree(leaves8);
        Check(tree8.Root() == 955777637L, "root of the 8-record log");
        for (int i = 0; i < 8; i++)
        {
            Check(tree8.Proof(i).Count == 3, "binary size audit path length");
            Check(MerkleTree.Verify(tree8.Root(), leaves8[i], tree8.Proof(i)), "n=8 proof " + i);
        }
        Check(!MerkleTree.Verify(tree8.Root(), "tx0!", tree8.Proof(0)), "tampered leaf in n=8");
        Check(tree8.Root() != tree.Root(), "different logs, different roots");

        var single = new MerkleTree(new List<string> { "only" });
        Check(single.Root() == MerkleTree.HashLeaf("only"), "single record root");
        Check(single.Proof(0).Count == 0, "single record empty proof");
        Check(MerkleTree.Verify(single.Root(), "only", single.Proof(0)), "single record verify");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
