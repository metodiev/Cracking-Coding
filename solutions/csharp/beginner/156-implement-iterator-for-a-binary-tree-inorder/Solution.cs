// Task 156 - Implement iterator for a binary tree (inorder). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class TreeNode
{
    public int Val;
    public TreeNode? Left, Right;
    public TreeNode(int v) { Val = v; }
}

/// <summary>
/// In-order iterator with an explicit stack: push the left spine of the current
/// subtree; Next() pops one node, then descends that node's right subtree's left
/// spine. O(1) amortised per step, O(h) extra memory.
/// </summary>
public class InOrderIterator
{
    private readonly Stack<TreeNode> stack = new Stack<TreeNode>();

    public InOrderIterator(TreeNode? root) { PushLeftSpine(root); }

    private void PushLeftSpine(TreeNode? node)
    {
        while (node != null)
        {
            stack.Push(node);
            node = node.Left;
        }
    }

    public bool HasNext() => stack.Count > 0;

    /// <summary>Next in-order value; throws InvalidOperationException once exhausted.</summary>
    public int Next()
    {
        if (stack.Count == 0) throw new InvalidOperationException("iterator exhausted");
        var node = stack.Pop();
        PushLeftSpine(node.Right);
        return node.Val;
    }
}

public static class Solution
{
    /// <summary>Build a binary tree from level-order values (null marks a missing child).</summary>
    public static TreeNode? BuildTree(params int?[] values)
    {
        if (values.Length == 0 || values[0] == null) return null;
        var root = new TreeNode(values[0]!.Value);
        var q = new Queue<TreeNode>();
        q.Enqueue(root);
        int i = 1;
        while (q.Count > 0 && i < values.Length)
        {
            var node = q.Dequeue();
            if (i < values.Length && values[i] != null)
            {
                node.Left = new TreeNode(values[i]!.Value);
                q.Enqueue(node.Left);
            }
            i++;
            if (i < values.Length && values[i] != null)
            {
                node.Right = new TreeNode(values[i]!.Value);
                q.Enqueue(node.Right);
            }
            i++;
        }
        return root;
    }

    public static void Main()
    {
        // valid BST: in-order must be ascending
        var bst = BuildTree(8, 3, 10, 1, 6, null, 14, null, null, 4, 7, 13);
        var it = new InOrderIterator(bst);
        var values = new List<int>();
        while (it.HasNext()) values.Add(it.Next());
        Check(string.Join(",", values) == "1,3,4,6,7,8,10,13,14", "BST in-order");
        for (int i = 1; i < values.Count; i++) Check(values[i - 1] < values[i], "BST ascending");

        // exhaustion: HasNext() is false and Next() throws
        Check(!it.HasNext(), "exhausted");
        bool threw = false;
        try { it.Next(); } catch (InvalidOperationException) { threw = true; }
        Check(threw, "next after exhaustion throws");

        // empty tree is immediately exhausted
        var empty = new InOrderIterator(null);
        Check(!empty.HasNext(), "empty exhausted");
        threw = false;
        try { empty.Next(); } catch (InvalidOperationException) { threw = true; }
        Check(threw, "empty next throws");

        // HasNext is idempotent (peeks without consuming) and works one step at a time
        var it2 = new InOrderIterator(BuildTree(1, 2, 3));
        Check(it2.HasNext() && it2.HasNext(), "hasNext idempotent");
        Check(it2.Next() == 2 && it2.Next() == 1, "left-most then root");
        Check(it2.HasNext() && it2.Next() == 3 && !it2.HasNext(), "right child last");

        // single node
        var it3 = new InOrderIterator(BuildTree(5));
        Check(it3.HasNext() && it3.Next() == 5 && !it3.HasNext(), "single node");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
