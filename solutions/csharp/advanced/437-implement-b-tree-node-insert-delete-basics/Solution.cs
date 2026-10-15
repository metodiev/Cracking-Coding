// Task 437 - Implement B-tree node insert/delete basics. (Tier: Advanced, CTCI Ch 14)
//
// A B-tree of order m = 4 (minimum degree t = 2): every node holds at most 2t-1 = 3 keys and, when
// internal, 2..4 children; non-root nodes keep at least t-1 = 1 key, and all leaves sit at the same
// depth. Multiset semantics are obtained by storing collision-free composite keys
// key = (value << 20) | serial: each insertion gets a fresh serial, so the textbook unique-key
// insert/delete apply unchanged while the sorted traversal still yields a sorted multiset of values.
// Insert uses the preemptive split rule - a full node (3 keys) is split around its median before
// descending, the median rises into the parent and the upper half becomes a new right sibling.
// Delete follows CLRS: a key found in an internal node is replaced by its in-order predecessor or
// successor; before descending into a child that holds only t-1 keys the child is repaired by
// borrowing a key from a sibling (rotation through the parent) or by merging with a sibling - the
// documented merge/borrow counter is asserted to be positive after the LCG workload. The sorted
// traversal is compared against a sorted multiset oracle on LCG operation scripts, and structure
// invariants (key order, occupancy, uniform leaf depth) are re-validated throughout.
using System;
using System.Collections.Generic;

internal class BTreeNode
{
    public readonly List<int> Keys = new List<int>();
    public readonly List<BTreeNode> Children = new List<BTreeNode>();
}

internal class BTree
{
    private const int T = 2;
    private const int Shift = 20;

    private BTreeNode root = new BTreeNode();
    public int Size;
    public int Splits;
    public int Merges;
    public int Borrows;
    private int nextSerial;

    public static int LowerBound(List<int> keys, int key)
    {
        int lo = 0;
        int hi = keys.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (keys[mid] < key) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    public void Insert(int value)
    {
        int key = (value << Shift) | nextSerial;
        nextSerial++;
        if (root.Keys.Count == 2 * T - 1)
        {
            var newRoot = new BTreeNode();
            newRoot.Children.Add(root);
            SplitChild(newRoot, 0);
            root = newRoot;
        }
        InsertNonfull(root, key);
        Size++;
    }

    private void SplitChild(BTreeNode parent, int index)
    {
        BTreeNode node = parent.Children[index];
        var right = new BTreeNode();
        int median = node.Keys[T - 1];
        right.Keys.AddRange(node.Keys.GetRange(T, node.Keys.Count - T));
        node.Keys.RemoveRange(T - 1, node.Keys.Count - (T - 1));
        if (node.Children.Count > 0)
        {
            right.Children.AddRange(node.Children.GetRange(T, node.Children.Count - T));
            node.Children.RemoveRange(T, node.Children.Count - T);
        }
        parent.Keys.Insert(index, median);
        parent.Children.Insert(index + 1, right);
        Splits++;
    }

    private void InsertNonfull(BTreeNode node, int key)
    {
        int index = LowerBound(node.Keys, key);
        if (node.Children.Count > 0)
        {
            if (node.Children[index].Keys.Count == 2 * T - 1)
            {
                SplitChild(node, index);
                if (key > node.Keys[index]) index++;
            }
            InsertNonfull(node.Children[index], key);
        }
        else
        {
            node.Keys.Insert(index, key);
        }
    }

    private int? LowerBoundKey(BTreeNode node, int baseKey)
    {
        int index = LowerBound(node.Keys, baseKey);
        if (node.Children.Count > 0)
        {
            int? found = LowerBoundKey(node.Children[index], baseKey);
            if (found != null) return found;
            if (index < node.Keys.Count) return node.Keys[index];
            return null;
        }
        return index < node.Keys.Count ? node.Keys[index] : (int?)null;
    }

    public bool Contains(int value)
    {
        int? key = LowerBoundKey(root, value << Shift);
        return key != null && (key.Value >> Shift) == value;
    }

    public bool Delete(int value)
    {
        int? key = LowerBoundKey(root, value << Shift);
        if (key == null || (key.Value >> Shift) != value) return false;
        Delete(root, key.Value);
        if (root.Keys.Count == 0 && root.Children.Count > 0) root = root.Children[0];
        Size--;
        return true;
    }

    private void Delete(BTreeNode node, int key)
    {
        int index = LowerBound(node.Keys, key);
        if (index < node.Keys.Count && node.Keys[index] == key)
        {
            if (node.Children.Count == 0) node.Keys.RemoveAt(index);
            else DeleteInternal(node, index);
            return;
        }
        if (node.Children.Count == 0) return;
        bool flag = index == node.Keys.Count;
        int childIndex = index;
        if (node.Children[childIndex].Keys.Count == T - 1)
        {
            Fill(node, childIndex);
            if (flag && childIndex > node.Keys.Count) childIndex--;
        }
        Delete(node.Children[childIndex], key);
    }

    private void DeleteInternal(BTreeNode node, int index)
    {
        int key = node.Keys[index];
        BTreeNode left = node.Children[index];
        BTreeNode right = node.Children[index + 1];
        if (left.Keys.Count > T - 1)
        {
            int predecessor = MaxKey(left);
            node.Keys[index] = predecessor;
            Delete(left, predecessor);
        }
        else if (right.Keys.Count > T - 1)
        {
            int successor = MinKey(right);
            node.Keys[index] = successor;
            Delete(right, successor);
        }
        else
        {
            Merge(node, index);
            Delete(node.Children[index], key);
        }
    }

    private void Fill(BTreeNode node, int index)
    {
        if (index > 0 && node.Children[index - 1].Keys.Count > T - 1)
        {
            BorrowFromLeft(node, index);
        }
        else if (index < node.Keys.Count && node.Children[index + 1].Keys.Count > T - 1)
        {
            BorrowFromRight(node, index);
        }
        else if (index < node.Keys.Count)
        {
            Merge(node, index);
        }
        else
        {
            Merge(node, index - 1);
        }
    }

    private void BorrowFromLeft(BTreeNode node, int index)
    {
        BTreeNode child = node.Children[index];
        BTreeNode left = node.Children[index - 1];
        child.Keys.Insert(0, node.Keys[index - 1]);
        node.Keys[index - 1] = left.Keys[left.Keys.Count - 1];
        left.Keys.RemoveAt(left.Keys.Count - 1);
        if (left.Children.Count > 0)
        {
            child.Children.Insert(0, left.Children[left.Children.Count - 1]);
            left.Children.RemoveAt(left.Children.Count - 1);
        }
        Borrows++;
    }

    private void BorrowFromRight(BTreeNode node, int index)
    {
        BTreeNode child = node.Children[index];
        BTreeNode right = node.Children[index + 1];
        child.Keys.Add(node.Keys[index]);
        node.Keys[index] = right.Keys[0];
        right.Keys.RemoveAt(0);
        if (right.Children.Count > 0)
        {
            child.Children.Add(right.Children[0]);
            right.Children.RemoveAt(0);
        }
        Borrows++;
    }

    private void Merge(BTreeNode node, int index)
    {
        BTreeNode left = node.Children[index];
        BTreeNode right = node.Children[index + 1];
        left.Keys.Add(node.Keys[index]);
        node.Keys.RemoveAt(index);
        left.Keys.AddRange(right.Keys);
        if (left.Children.Count > 0) left.Children.AddRange(right.Children);
        node.Children.RemoveAt(index + 1);
        Merges++;
    }

    private static int MinKey(BTreeNode node)
    {
        while (node.Children.Count > 0) node = node.Children[0];
        return node.Keys[0];
    }

    private static int MaxKey(BTreeNode node)
    {
        while (node.Children.Count > 0) node = node.Children[node.Children.Count - 1];
        return node.Keys[node.Keys.Count - 1];
    }

    public List<int> InorderValues()
    {
        var outList = new List<int>();
        Inorder(root, outList);
        return outList;
    }

    private static void Inorder(BTreeNode node, List<int> outList)
    {
        for (int i = 0; i < node.Keys.Count; i++)
        {
            if (node.Children.Count > 0) Inorder(node.Children[i], outList);
            outList.Add(node.Keys[i] >> Shift);
        }
        if (node.Children.Count > 0) Inorder(node.Children[node.Keys.Count], outList);
    }

    public bool Validate() => Validate(root, true, 0, -1) >= 0;

    private static int Validate(BTreeNode node, bool isRoot, int depth, int leafDepth)
    {
        if (node.Keys.Count > 2 * T - 1) return -1;
        for (int i = 1; i < node.Keys.Count; i++)
        {
            if (node.Keys[i - 1] >= node.Keys[i]) return -1;
        }
        if (node.Children.Count > 0)
        {
            if (node.Children.Count != node.Keys.Count + 1) return -1;
            if (node.Children.Count < (isRoot ? 2 : T)) return -1;
            foreach (BTreeNode child in node.Children)
            {
                leafDepth = Validate(child, false, depth + 1, leafDepth);
                if (leafDepth < 0) return -1;
            }
            return leafDepth;
        }
        if (!isRoot && node.Keys.Count < T - 1) return -1;
        if (leafDepth != -1 && leafDepth != depth) return -1;
        return depth;
    }
}

public static class Solution
{
    private static void SortedInsert(List<int> oracle, int value)
    {
        oracle.Insert(BTree.LowerBound(oracle, value), value);
    }

    public static void Main()
    {
        var tree = new BTree();
        Check(tree.Validate(), "empty validate");
        Check(tree.InorderValues().Count == 0, "empty order");
        Check(!tree.Delete(5), "delete from empty");

        foreach (int value in new[] { 10, 20, 30, 40 }) tree.Insert(value);
        Check(tree.Splits >= 1, "root split " + tree.Splits);
        Check(string.Join(",", tree.InorderValues()) == "10,20,30,40", "hand order");
        Check(tree.Validate(), "hand validate");
        Check(tree.Contains(30) && !tree.Contains(35), "hand contains");
        Check(tree.Size == 4, "hand size");

        tree.Insert(10);
        tree.Insert(10);
        Check(string.Join(",", tree.InorderValues()) == "10,10,10,20,30,40", "duplicates");
        Check(tree.Delete(10), "delete duplicate");
        Check(string.Join(",", tree.InorderValues()) == "10,10,20,30,40", "after one delete");
        Check(tree.Delete(10) && tree.Delete(10), "delete remaining duplicates");
        Check(!tree.Contains(10), "no 10 left");
        Check(!tree.Delete(10), "delete absent 10");
        Check(string.Join(",", tree.InorderValues()) == "20,30,40", "after duplicate drain");

        foreach (int value in new[] { 35, 25, 15, 5, 45, 50, 55, 60 }) tree.Insert(value);
        Check(string.Join(",", tree.InorderValues()) == "5,15,20,25,30,35,40,45,50,55,60", "second hand order");
        Check(tree.Validate(), "second hand validate");
        foreach (int value in new[] { 30, 5, 60, 20, 45, 15, 55, 25, 35, 50, 40 })
        {
            Check(tree.Delete(value), "scripted hand delete");
        }
        Check(tree.InorderValues().Count == 0, "hand drained");
        Check(tree.Size == 0, "hand drained size");
        Check(tree.Validate(), "hand drained validate");

        tree = new BTree();
        var oracle = new List<int>();
        int seed = 4370;
        for (int step = 0; step < 300; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 4;
            seed = (seed * 75 + 74) % 65537;
            if (op <= 1)
            {
                int value = seed % 30;
                tree.Insert(value);
                SortedInsert(oracle, value);
            }
            else if (op == 2 && oracle.Count > 0)
            {
                int index = seed % oracle.Count;
                int value = oracle[index];
                oracle.RemoveAt(index);
                Check(tree.Delete(value), "random delete step " + step);
                bool stillPresent = oracle.Contains(value);
                Check(tree.Delete(value) == stillPresent, "random second delete step " + step);
                if (stillPresent) oracle.Remove(value);
            }
            else
            {
                int value = 100 + seed % 5;
                Check(!tree.Delete(value), "random absent step " + step);
            }
            Check(tree.Size == oracle.Count, "random size step " + step);
            if (step % 20 == 0 || step == 299)
            {
                Check(string.Join(",", tree.InorderValues()) == string.Join(",", oracle), "random order step " + step);
                Check(tree.Validate(), "random validate step " + step);
            }
        }
        Check(oracle.Count > 0, "workload left values");
        Check(string.Join(",", tree.InorderValues()) == string.Join(",", oracle), "final order");
        Check(tree.Validate(), "final validate");
        Check(tree.Splits > 0 && tree.Merges > 0, "counters " + tree.Splits + "/" + tree.Merges + "/" + tree.Borrows);

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
