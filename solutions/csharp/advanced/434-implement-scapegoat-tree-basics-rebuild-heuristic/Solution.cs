// Task 434 - Implement scapegoat tree basics (rebuild heuristic). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A scapegoat tree is an unbalanced BST that stays near-balanced by whole-subtree rebuilds instead
// of rotations. Documented constants: balance factor alpha = 0.75 (a node may not have a child
// subtree larger than alpha * its own size) and the delete trigger size < alpha * max_size, where
// max_size is the peak size since the last full rebuild. Insert walks down, then walks back up
// updating cached sizes and picks the deepest ancestor whose child exceeds alpha * its size - the
// scapegoat - and rebuilds its subtree into a perfectly balanced half-split. Delete is physical:
// the key is replaced by its in-order successor and sizes are repaired on the way up; falling below
// alpha * max_size rebuilds the whole tree. Keys are unique (duplicate inserts are no-ops), matching
// the sorted-list oracle used on LCG operation scripts. Rebuilds are counted: the adversarial
// ascending insert 1..200 triggers at least one rebuild (a final assert documents the count), and
// every state is checked for BST order, cached sizes and O(log n) height.
using System;
using System.Collections.Generic;

internal class ScapegoatNode
{
    public int Key;
    public ScapegoatNode? Left;
    public ScapegoatNode? Right;
    public int SubtreeSize;

    public ScapegoatNode(int key)
    {
        Key = key;
        SubtreeSize = 1;
    }
}

internal class ScapegoatTree
{
    private readonly double alpha;
    private ScapegoatNode? root;
    private int total;
    private int maxSize;
    public int Rebuilds;

    public ScapegoatTree() : this(0.75) { }

    public ScapegoatTree(double alpha)
    {
        this.alpha = alpha;
    }

    public int Size() => total;

    public bool Contains(int key)
    {
        ScapegoatNode? node = root;
        while (node != null)
        {
            if (key == node.Key) return true;
            node = key < node.Key ? node.Left : node.Right;
        }
        return false;
    }

    public bool Insert(int key)
    {
        if (root == null)
        {
            root = new ScapegoatNode(key);
            total = 1;
            maxSize = 1;
            return true;
        }
        var path = new List<ScapegoatNode>();
        ScapegoatNode node = root;
        while (true)
        {
            path.Add(node);
            if (key == node.Key) return false;
            if (key < node.Key)
            {
                if (node.Left == null)
                {
                    node.Left = new ScapegoatNode(key);
                    break;
                }
                node = node.Left;
            }
            else
            {
                if (node.Right == null)
                {
                    node.Right = new ScapegoatNode(key);
                    break;
                }
                node = node.Right;
            }
        }
        int scapegoatIndex = -1;
        for (int i = path.Count - 1; i >= 0; i--)
        {
            ScapegoatNode current = path[i];
            current.SubtreeSize++;
            if (scapegoatIndex == -1)
            {
                int left = current.Left == null ? 0 : current.Left.SubtreeSize;
                int right = current.Right == null ? 0 : current.Right.SubtreeSize;
                if (left > alpha * current.SubtreeSize || right > alpha * current.SubtreeSize)
                {
                    scapegoatIndex = i;
                }
            }
        }
        total++;
        maxSize = Math.Max(maxSize, total);
        if (scapegoatIndex >= 0) RebuildAt(path, scapegoatIndex);
        return true;
    }

    public bool Delete(int key)
    {
        if (!Contains(key)) return false;
        root = Delete(root, key);
        total--;
        if (total < alpha * maxSize)
        {
            List<int> keys = Collect(root);
            root = Build(keys, 0, keys.Count - 1);
            maxSize = total;
            Rebuilds++;
        }
        return true;
    }

    private ScapegoatNode? Delete(ScapegoatNode? node, int key)
    {
        if (key < node!.Key) node.Left = Delete(node.Left, key);
        else if (key > node.Key) node.Right = Delete(node.Right, key);
        else return DeleteNode(node);
        Update(node);
        return node;
    }

    private ScapegoatNode? DeleteNode(ScapegoatNode node)
    {
        if (node.Left == null) return node.Right;
        if (node.Right == null) return node.Left;
        ScapegoatNode successor = node.Right;
        while (successor.Left != null) successor = successor.Left;
        node.Key = successor.Key;
        node.Right = Delete(node.Right, successor.Key);
        Update(node);
        return node;
    }

    public List<int> Inorder() => Collect(root);

    public int Height() => Height(root);

    private static int Height(ScapegoatNode? node)
    {
        if (node == null) return 0;
        return 1 + Math.Max(Height(node.Left), Height(node.Right));
    }

    public bool Validate() => Validate(root, long.MinValue, long.MaxValue) == total;

    private static int Validate(ScapegoatNode? node, long low, long high)
    {
        if (node == null) return 0;
        if (!(low < node.Key && node.Key < high)) return -1;
        int leftSize = Validate(node.Left, low, node.Key);
        int rightSize = Validate(node.Right, node.Key, high);
        if (leftSize < 0 || rightSize < 0) return -1;
        if (node.SubtreeSize != 1 + leftSize + rightSize) return -1;
        return node.SubtreeSize;
    }

    private static List<int> Collect(ScapegoatNode? node)
    {
        var outList = new List<int>();
        var stack = new Stack<ScapegoatNode>();
        while (stack.Count > 0 || node != null)
        {
            while (node != null)
            {
                stack.Push(node);
                node = node.Left;
            }
            node = stack.Pop();
            outList.Add(node.Key);
            node = node.Right;
        }
        return outList;
    }

    private static void Update(ScapegoatNode node)
    {
        node.SubtreeSize = 1 + (node.Left == null ? 0 : node.Left.SubtreeSize)
                             + (node.Right == null ? 0 : node.Right.SubtreeSize);
    }

    private void RebuildAt(List<ScapegoatNode> path, int index)
    {
        ScapegoatNode node = path[index];
        ScapegoatNode? rebuilt = Build(Collect(node), 0, node.SubtreeSize - 1);
        if (index == 0)
        {
            root = rebuilt;
        }
        else
        {
            ScapegoatNode parent = path[index - 1];
            if (parent.Left == node) parent.Left = rebuilt;
            else parent.Right = rebuilt;
        }
        Rebuilds++;
    }

    private static ScapegoatNode? Build(List<int> keys, int lo, int hi)
    {
        if (lo > hi) return null;
        int mid = (lo + hi) / 2;
        var node = new ScapegoatNode(keys[mid]);
        node.Left = Build(keys, lo, mid - 1);
        node.Right = Build(keys, mid + 1, hi);
        Update(node);
        return node;
    }
}

public static class Solution
{
    private static long HeightBound(int size)
    {
        return 3L * (long)Math.Ceiling(Math.Log(size + 1) / Math.Log(2)) + 3;
    }

    public static void Main()
    {
        var tree = new ScapegoatTree();
        Check(tree.Size() == 0, "empty size");
        Check(tree.Inorder().Count == 0, "empty order");
        Check(tree.Validate(), "empty validate");
        Check(!tree.Delete(0), "delete from empty");

        foreach (int key in new[] { 10, 5, 15, 3 }) Check(tree.Insert(key), "hand insert");
        Check(!tree.Insert(10), "duplicate insert");
        Check(string.Join(",", tree.Inorder()) == "3,5,10,15", "hand order");
        Check(tree.Contains(15) && !tree.Contains(4), "contains");
        Check(tree.Validate(), "hand validate");
        Check(tree.Delete(5), "hand delete");
        Check(!tree.Delete(5), "repeat delete");
        Check(string.Join(",", tree.Inorder()) == "3,10,15", "order after delete");
        Check(tree.Validate(), "validate after delete");

        tree = new ScapegoatTree();
        for (int key = 1; key <= 200; key++) tree.Insert(key);
        var ascending = new List<int>();
        for (int key = 1; key <= 200; key++) ascending.Add(key);
        Check(string.Join(",", tree.Inorder()) == string.Join(",", ascending), "ascending order");
        Check(tree.Rebuilds >= 1, "ascending rebuilds " + tree.Rebuilds);
        Check(tree.Height() <= HeightBound(tree.Size()), "ascending height " + tree.Height());
        for (int key = 2; key <= 200; key += 2) Check(tree.Delete(key), "ascending delete");
        var odds = new List<int>();
        for (int key = 1; key <= 200; key += 2) odds.Add(key);
        Check(string.Join(",", tree.Inorder()) == string.Join(",", odds), "odds order");
        Check(tree.Validate(), "odds validate");
        for (int key = 1; key <= 200; key += 2) Check(tree.Delete(key), "drain delete");
        Check(tree.Size() == 0 && tree.Inorder().Count == 0, "drained");
        Check(tree.Validate(), "drained validate");

        tree = new ScapegoatTree();
        var oracle = new List<int>();
        int seed = 4340;
        for (int step = 0; step < 400; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 3;
            seed = (seed * 75 + 74) % 65537;
            int key = seed % 100;
            if (op == 0)
            {
                bool got = tree.Insert(key);
                bool expected = !oracle.Contains(key);
                if (expected)
                {
                    int index = oracle.BinarySearch(key);
                    oracle.Insert(index < 0 ? ~index : index, key);
                }
                Check(got == expected, "random insert step " + step);
            }
            else
            {
                bool got = tree.Delete(key);
                bool expected = oracle.Contains(key);
                if (expected) oracle.Remove(key);
                Check(got == expected, "random delete step " + step);
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
        Check(tree.Height() <= HeightBound(tree.Size()), "final height " + tree.Height());

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
