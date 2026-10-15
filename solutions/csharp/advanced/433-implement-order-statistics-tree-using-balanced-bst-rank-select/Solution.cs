// Task 433 - Implement order statistics tree using balanced BST (rank/select). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// The balanced BST is a treap: a BST ordered by key plus a min-heap ordered by priority, kept
// balanced by rotations on insert and by merging the two children on delete. Each node caches the
// size of its subtree and a multiplicity count, so it acts as a multiset: Rank(x) returns the
// number of stored elements strictly below x and Select(k) returns the k-th smallest (1-based,
// counting multiplicities). Priorities are drawn from the shared LCG stream, which makes the tree
// shape - and therefore every assertion - fully deterministic. Assertions compare insert/delete/
// rank/select/count against a sorted-list oracle on LCG operation scripts and re-check the
// heap/BST/size invariants with a structural validator.
using System;
using System.Collections.Generic;

internal class TreapNode
{
    public int Key;
    public int Priority;
    public int Count;
    public int SubtreeSize;
    public TreapNode? Left;
    public TreapNode? Right;

    public TreapNode(int key, int priority)
    {
        Key = key;
        Priority = priority;
        Count = 1;
        SubtreeSize = 1;
    }
}

internal class OrderStatisticsTree
{
    private TreapNode? root;
    private int total;
    private bool removed;

    public int Size() => total;

    private static int SizeOf(TreapNode? node) => node == null ? 0 : node.SubtreeSize;

    private static void Update(TreapNode node)
    {
        node.SubtreeSize = SizeOf(node.Left) + node.Count + SizeOf(node.Right);
    }

    private TreapNode RotateRight(TreapNode node)
    {
        TreapNode child = node.Left!;
        node.Left = child.Right;
        child.Right = node;
        Update(node);
        Update(child);
        return child;
    }

    private TreapNode RotateLeft(TreapNode node)
    {
        TreapNode child = node.Right!;
        node.Right = child.Left;
        child.Left = node;
        Update(node);
        Update(child);
        return child;
    }

    public void Insert(int key, int priority)
    {
        root = Insert(root, key, priority);
        total++;
    }

    private TreapNode Insert(TreapNode? node, int key, int priority)
    {
        if (node == null) return new TreapNode(key, priority);
        if (key == node.Key)
        {
            node.Count++;
            node.SubtreeSize++;
            return node;
        }
        if (key < node.Key)
        {
            node.Left = Insert(node.Left, key, priority);
            if (node.Left.Priority < node.Priority) return RotateRight(node);
        }
        else
        {
            node.Right = Insert(node.Right, key, priority);
            if (node.Right.Priority < node.Priority) return RotateLeft(node);
        }
        Update(node);
        return node;
    }

    public bool Delete(int key)
    {
        removed = false;
        root = Delete(root, key);
        if (removed) total--;
        return removed;
    }

    private TreapNode? Delete(TreapNode? node, int key)
    {
        if (node == null) return null;
        if (key == node.Key)
        {
            removed = true;
            if (node.Count > 1)
            {
                node.Count--;
                node.SubtreeSize--;
                return node;
            }
            return Merge(node.Left, node.Right);
        }
        if (key < node.Key) node.Left = Delete(node.Left, key);
        else node.Right = Delete(node.Right, key);
        Update(node);
        return node;
    }

    private TreapNode? Merge(TreapNode? left, TreapNode? right)
    {
        if (left == null) return right;
        if (right == null) return left;
        if (left.Priority <= right.Priority)
        {
            left.Right = Merge(left.Right, right);
            Update(left);
            return left;
        }
        right.Left = Merge(left, right.Left);
        Update(right);
        return right;
    }

    public int Rank(int key) => Rank(root, key);

    private int Rank(TreapNode? node, int key)
    {
        if (node == null) return 0;
        if (key <= node.Key) return Rank(node.Left, key);
        return SizeOf(node.Left) + node.Count + Rank(node.Right, key);
    }

    public int Select(int k)
    {
        if (k < 1 || k > total) throw new IndexOutOfRangeException("select index out of range");
        return Select(root!, k);
    }

    private int Select(TreapNode node, int k)
    {
        int leftSize = SizeOf(node.Left);
        if (k <= leftSize) return Select(node.Left!, k);
        if (k <= leftSize + node.Count) return node.Key;
        return Select(node.Right!, k - leftSize - node.Count);
    }

    public int CountOf(int key)
    {
        TreapNode? node = root;
        while (node != null)
        {
            if (key == node.Key) return node.Count;
            node = key < node.Key ? node.Left : node.Right;
        }
        return 0;
    }

    public List<int> Inorder()
    {
        var outList = new List<int>();
        Inorder(root, outList);
        return outList;
    }

    private static void Inorder(TreapNode? node, List<int> outList)
    {
        if (node == null) return;
        Inorder(node.Left, outList);
        for (int i = 0; i < node.Count; i++) outList.Add(node.Key);
        Inorder(node.Right, outList);
    }

    public bool Validate() => Validate(root, long.MinValue, long.MaxValue) == total;

    private static int Validate(TreapNode? node, long low, long high)
    {
        if (node == null) return 0;
        if (!(low < node.Key && node.Key < high)) return -1;
        int leftSize = Validate(node.Left, low, node.Key);
        int rightSize = Validate(node.Right, node.Key, high);
        if (leftSize < 0 || rightSize < 0) return -1;
        if (node.SubtreeSize != leftSize + node.Count + rightSize) return -1;
        if (node.Left != null && node.Left.Priority < node.Priority) return -1;
        if (node.Right != null && node.Right.Priority < node.Priority) return -1;
        return node.SubtreeSize;
    }
}

public static class Solution
{
    private static void SortedInsert(List<int> sortedValues, int value)
    {
        int index = sortedValues.BinarySearch(value);
        if (index < 0) index = ~index;
        sortedValues.Insert(index, value);
    }

    private static int LowerBound(List<int> sortedValues, int value)
    {
        int lo = 0;
        int hi = sortedValues.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (sortedValues[mid] < value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    public static void Main()
    {
        var tree = new OrderStatisticsTree();
        Check(tree.Size() == 0, "empty size");
        var seeded = new[] { new[] { 5, 100 }, new[] { 1, 90 }, new[] { 9, 80 }, new[] { 3, 70 }, new[] { 3, 110 } };
        foreach (var pair in seeded) tree.Insert(pair[0], pair[1]);
        Check(string.Join(",", tree.Inorder()) == "1,3,3,5,9", "hand inorder");
        Check(tree.Size() == 5, "hand size");
        Check(tree.Validate(), "hand validate");
        Check(tree.Select(1) == 1 && tree.Select(2) == 3 && tree.Select(3) == 3, "hand select low");
        Check(tree.Select(4) == 5 && tree.Select(5) == 9, "hand select high");
        Check(tree.Rank(1) == 0 && tree.Rank(3) == 1, "hand rank low");
        Check(tree.Rank(4) == 3 && tree.Rank(9) == 4, "hand rank high");
        Check(tree.CountOf(3) == 2 && tree.CountOf(4) == 0, "hand counts");
        Check(tree.Delete(3), "hand delete");
        Check(string.Join(",", tree.Inorder()) == "1,3,5,9", "inorder after delete");
        Check(tree.Select(3) == 5, "select after delete");
        Check(!tree.Delete(100), "absent delete");
        Check(tree.Validate(), "validate after delete");

        tree = new OrderStatisticsTree();
        var oracle = new List<int>();
        int seed = 4331;
        for (int step = 0; step < 400; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 5;
            if (op <= 1)
            {
                seed = (seed * 75 + 74) % 65537;
                int key = seed % 60;
                seed = (seed * 75 + 74) % 65537;
                tree.Insert(key, seed);
                SortedInsert(oracle, key);
            }
            else if (op == 2)
            {
                seed = (seed * 75 + 74) % 65537;
                int key = seed % 60;
                bool expected = oracle.Contains(key);
                if (expected) oracle.Remove(key);
                Check(tree.Delete(key) == expected, "random delete step " + step);
            }
            else if (op == 3)
            {
                seed = (seed * 75 + 74) % 65537;
                int key = seed % 60;
                int count = 0;
                foreach (int value in oracle)
                {
                    if (value == key) count++;
                }
                Check(tree.Rank(key) == LowerBound(oracle, key), "random rank step " + step);
                Check(tree.CountOf(key) == count, "random count step " + step);
            }
            else if (oracle.Count > 0)
            {
                seed = (seed * 75 + 74) % 65537;
                int k = 1 + seed % oracle.Count;
                Check(tree.Select(k) == oracle[k - 1], "random select step " + step);
            }
            Check(tree.Size() == oracle.Count, "random size step " + step);
            if (step % 25 == 0)
            {
                Check(string.Join(",", tree.Inorder()) == string.Join(",", oracle), "random inorder step " + step);
                Check(tree.Validate(), "random validate step " + step);
            }
        }
        Check(oracle.Count > 0, "workload left values");
        Check(string.Join(",", tree.Inorder()) == string.Join(",", oracle), "final inorder");
        Check(tree.Validate(), "final validate");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
