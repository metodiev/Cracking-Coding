// Task 477 - Implement B+ tree leaf-range query and insert basics. (Tier: Advanced, CTCI Ch 14 — Databases)
using System;
using System.Collections.Generic;

internal class BNode
{
    public readonly bool IsLeaf;
    public readonly List<int> Keys = new List<int>();
    public readonly List<BNode> Children = new List<BNode>(); // internal nodes only
    public BNode Next;                                        // leaf nodes only

    public BNode(bool isLeaf) => IsLeaf = isLeaf;
}

internal class Split
{
    public readonly int Sep;
    public readonly BNode Right;

    public Split(int sep, BNode right)
    {
        Sep = sep;
        Right = right;
    }
}

internal class BPlusTree
{
    public readonly int MaxKeys;
    public BNode Root = new BNode(true);
    public int Size;

    public BPlusTree(int maxKeys) => MaxKeys = maxKeys;

    public BNode LeafFor(int key)
    {
        BNode node = Root;
        while (!node.IsLeaf)
        {
            int idx = 0;
            while (idx < node.Keys.Count && node.Keys[idx] <= key) idx++;
            node = node.Children[idx];
        }
        return node;
    }

    public bool Contains(int key) => LeafFor(key).Keys.Contains(key);

    public List<int> AllKeys()
    {
        BNode leaf = LeafFor(int.MinValue);
        var output = new List<int>();
        while (leaf != null)
        {
            output.AddRange(leaf.Keys);
            leaf = leaf.Next;
        }
        return output;
    }

    public List<int> Range(int lo, int hi)
    {
        var output = new List<int>();
        if (lo > hi) return output;
        BNode leaf = LeafFor(lo);
        while (leaf != null)
        {
            foreach (int key in leaf.Keys)
            {
                if (key > hi) return output;
                if (key >= lo) output.Add(key);
            }
            leaf = leaf.Next;
        }
        return output;
    }

    public bool Insert(int key)
    {
        if (Contains(key)) return false;
        Split res = InsertInto(Root, key);
        if (res != null)
        {
            var newRoot = new BNode(false);
            newRoot.Keys.Add(res.Sep);
            newRoot.Children.Add(Root);
            newRoot.Children.Add(res.Right);
            Root = newRoot;
        }
        Size++;
        return true;
    }

    private Split InsertInto(BNode node, int key)
    {
        if (node.IsLeaf)
        {
            int pos = 0;
            while (pos < node.Keys.Count && node.Keys[pos] < key) pos++;
            node.Keys.Insert(pos, key);
            if (node.Keys.Count <= MaxKeys) return null;
            int mid = node.Keys.Count / 2;
            var right = new BNode(true);
            right.Keys.AddRange(node.Keys.GetRange(mid, node.Keys.Count - mid));
            node.Keys.RemoveRange(mid, node.Keys.Count - mid);
            right.Next = node.Next;
            node.Next = right;
            return new Split(right.Keys[0], right);
        }
        int idx = 0;
        while (idx < node.Keys.Count && node.Keys[idx] <= key) idx++;
        Split res = InsertInto(node.Children[idx], key);
        if (res == null) return null;
        node.Keys.Insert(idx, res.Sep);
        node.Children.Insert(idx + 1, res.Right);
        if (node.Keys.Count <= MaxKeys) return null;
        int mid2 = node.Children.Count / 2;
        int promoted = node.Keys[mid2 - 1];
        var rightNode = new BNode(false);
        rightNode.Children.AddRange(node.Children.GetRange(mid2, node.Children.Count - mid2));
        rightNode.Keys.AddRange(node.Keys.GetRange(mid2, node.Keys.Count - mid2));
        node.Children.RemoveRange(mid2, node.Children.Count - mid2);
        node.Keys.RemoveRange(mid2 - 1, node.Keys.Count - (mid2 - 1));
        return new Split(promoted, rightNode);
    }
}

public static class Solution
{
    private static int Lcg(int seed) => (seed * 75 + 74) % 65537;

    private static int MinKey(BNode node)
    {
        while (!node.IsLeaf) node = node.Children[0];
        return node.Keys[0];
    }

    private static List<BNode> CollectLeaves(BNode node)
    {
        var output = new List<BNode>();
        if (node.IsLeaf)
        {
            output.Add(node);
            return output;
        }
        foreach (BNode child in node.Children) output.AddRange(CollectLeaves(child));
        return output;
    }

    private static List<BNode> ChainLeaves(BPlusTree tree)
    {
        BNode leaf = tree.LeafFor(int.MinValue);
        var output = new List<BNode>();
        while (leaf != null)
        {
            output.Add(leaf);
            leaf = leaf.Next;
        }
        return output;
    }

    private static void CheckInvariants(BPlusTree tree)
    {
        var depths = new List<int>();
        WalkInvariants(tree, tree.Root, 0, depths);
        Check(new HashSet<int>(depths).Count == 1, "all leaves at equal depth");
        List<BNode> byChain = ChainLeaves(tree);
        List<BNode> byDfs = CollectLeaves(tree.Root);
        Check(byChain.Count == byDfs.Count, "leaf chain covers every leaf");
        for (int i = 0; i < byChain.Count; i++)
        {
            Check(SameKeys(byChain[i].Keys, byDfs[i].Keys), "leaf chain order matches tree order");
        }
    }

    private static void WalkInvariants(BPlusTree tree, BNode node, int depth, List<int> depths)
    {
        var sorted = new List<int>(node.Keys);
        sorted.Sort();
        Check(SameKeys(node.Keys, sorted), "keys sorted");
        Check(node.Keys.Count <= tree.MaxKeys, "node capacity");
        if (node.IsLeaf)
        {
            depths.Add(depth);
            return;
        }
        Check(node.Children.Count == node.Keys.Count + 1, "children = keys + 1");
        for (int i = 0; i < node.Keys.Count; i++)
        {
            Check(MinKey(node.Children[i + 1]) == node.Keys[i], "separator equals right subtree minimum");
        }
        foreach (BNode child in node.Children) WalkInvariants(tree, child, depth + 1, depths);
    }

    private static bool SameKeys(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static int LeafCount(BPlusTree tree) => CollectLeaves(tree.Root).Count;

    private static int Height(BPlusTree tree)
    {
        int depth = 1;
        BNode node = tree.Root;
        while (!node.IsLeaf)
        {
            node = node.Children[0];
            depth++;
        }
        return depth;
    }

    private static List<int> Filter(List<int> values, int lo, int hi)
    {
        var output = new List<int>();
        foreach (int v in values) if (v >= lo && v <= hi) output.Add(v);
        return output;
    }

    public static void Main()
    {
        // Ascending inserts force a deterministic split cascade (hand-traced).
        var tree = new BPlusTree(3);
        var brute = new List<int>();
        int[][] probes = { new[] { 1, 12 }, new[] { 4, 9 }, new[] { 5, 5 }, new[] { 3, 3 }, new[] { 10, 20 }, new[] { 0, 0 }, new[] { 12, 12 }, new[] { 2, 11 } };
        for (int key = 1; key <= 12; key++)
        {
            Check(tree.Insert(key), "insert " + key);
            brute.Add(key);
            CheckInvariants(tree);
            foreach (int[] probe in probes)
            {
                Check(SameKeys(tree.Range(probe[0], probe[1]), Filter(brute, probe[0], probe[1])),
                    "range " + probe[0] + ".." + probe[1]);
            }
        }
        Check(SameKeys(tree.Root.Keys, new List<int> { 5 }), "root separator hand-traced to [5]");
        Check(LeafCount(tree) == 6, "six leaves");
        Check(Height(tree) == 3, "three levels");
        Check(SameKeys(tree.AllKeys(), brute), "all keys in order");
        Check(!tree.Insert(7), "duplicates are rejected");
        Check(SameKeys(tree.AllKeys(), brute) && tree.Size == 12, "duplicate left the tree unchanged");

        // Descending inserts exercise the left-splitting mirror image.
        var tree2 = new BPlusTree(3);
        var brute2 = new List<int>();
        for (int key = 12; key >= 1; key--)
        {
            Check(tree2.Insert(key), "insert " + key);
            brute2.Add(key);
            CheckInvariants(tree2);
            var sorted = new List<int>(brute2);
            sorted.Sort();
            Check(SameKeys(tree2.Range(1, 12), sorted), "descending range");
        }
        var oneToTwelve = new List<int>();
        for (int i = 1; i <= 12; i++) oneToTwelve.Add(i);
        Check(SameKeys(tree2.AllKeys(), oneToTwelve), "descending keys complete");

        // Randomized inserts and range scans against the sorted-list oracle.
        var tree3 = new BPlusTree(4);
        var inserted = new SortedSet<int>();
        int seed = 4242;
        for (int i = 0; i < 40; i++)
        {
            seed = Lcg(seed);
            int key = seed % 64;
            if (inserted.Contains(key))
            {
                Check(!tree3.Insert(key), "duplicate rejected");
            }
            else
            {
                Check(tree3.Insert(key), "insert " + key);
                inserted.Add(key);
            }
            CheckInvariants(tree3);
        }
        var keysSorted = new List<int>(inserted);
        Check(keysSorted.Count > 25, "random phase inserted enough keys");
        Check(SameKeys(tree3.AllKeys(), keysSorted), "all keys");
        for (int i = 0; i < 80; i++)
        {
            seed = Lcg(seed);
            int lo = seed % 70;
            seed = Lcg(seed);
            int hi = lo + seed % (70 - lo);
            Check(SameKeys(tree3.Range(lo, hi), Filter(keysSorted, lo, hi)), "random range " + lo + ".." + hi);
        }
        Check(tree3.Range(65, 63).Count == 0, "reversed range is empty");
        Check(tree3.Range(200, 300).Count == 0, "beyond max is empty");
        Check(SameKeys(tree3.Range(0, 0), inserted.Contains(0) ? new List<int> { 0 } : new List<int>()), "singleton range");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
