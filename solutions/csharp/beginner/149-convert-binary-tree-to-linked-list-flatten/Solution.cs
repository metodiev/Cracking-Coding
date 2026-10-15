// Task 149 - Convert binary tree to linked list (flatten). (Tier: Beginner, CTCI Ch 4)
//
// LeetCode 114: rewrite the tree IN PLACE into its preorder order, using the right
// pointers as the "next" links and leaving every left pointer empty.
// [1, 2, 5, 3, 4, null, 6] becomes the chain 1 -> 2 -> 3 -> 4 -> 5 -> 6.
using System;
using System.Collections.Generic;

public class TreeNode
{
    public int Val;
    public TreeNode? Left, Right;
    public TreeNode(int v) { Val = v; }
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

    /// <summary>Recursive: flatten both subtrees, then splice the left chain in front of the right one.</summary>
    public static TreeNode? Flatten(TreeNode? root)
    {
        if (root == null) return null;
        var left = root.Left;
        var right = root.Right;
        Flatten(left);
        Flatten(right);
        root.Left = null;
        root.Right = left;
        var node = root;
        while (node.Right != null) node = node.Right;   // tail of the flattened left part
        node.Right = right;
        return root;
    }

    /// <summary>Preorder with an explicit stack: link each node to the previously visited one.</summary>
    public static TreeNode? FlattenIterative(TreeNode? root)
    {
        if (root == null) return null;
        var stack = new Stack<TreeNode>();
        stack.Push(root);
        TreeNode? prev = null;
        while (stack.Count > 0)
        {
            var node = stack.Pop();
            if (prev != null)
            {
                prev.Left = null;
                prev.Right = node;
            }
            if (node.Right != null) stack.Push(node.Right);   // right first: left is visited first
            if (node.Left != null) stack.Push(node.Left);
            prev = node;
        }
        return root;
    }

    public static List<int> PreorderValues(TreeNode? root)
    {
        var values = new List<int>();
        PreorderInto(root, values);
        return values;
    }

    private static void PreorderInto(TreeNode? node, List<int> values)
    {
        if (node == null) return;
        values.Add(node.Val);
        PreorderInto(node.Left, values);
        PreorderInto(node.Right, values);
    }

    /// <summary>Values following the right pointers (the flattened preorder order).</summary>
    public static List<int> RightChain(TreeNode? root)
    {
        var values = new List<int>();
        while (root != null)
        {
            values.Add(root.Val);
            root = root.Right;
        }
        return values;
    }

    /// <summary>After flattening nothing may remain on a left pointer.</summary>
    public static bool AllLeftNull(TreeNode? node)
    {
        while (node != null)
        {
            if (node.Left != null) return false;
            node = node.Right;
        }
        return true;
    }

    public static void Main()
    {
        Check(Flatten(null) == null && FlattenIterative(null) == null, "empty tree");
        Check(string.Join(",", RightChain(Flatten(BuildTree(7)))) == "7", "single node");
        Check(string.Join(",", RightChain(Flatten(BuildTree(1, 2, 5, 3, 4, null, 6)))) == "1,2,3,4,5,6",
              "sample becomes the preorder chain");
        var cases = new[]
        {
            new int?[] { }, new int?[] { 7 }, new int?[] { 1, 2, 5, 3, 4, null, 6 },
            new int?[] { 1, 2, null, 3 }, new int?[] { 1, null, 2 },
            new int?[] { 1, 2, null, 3, null, 4 }, new int?[] { 1, 2, 3, 4, 5, null, 6 }
        };
        foreach (var data in cases)
        {
            string expected = string.Join(",", PreorderValues(BuildTree(data)));   // target order
            var recursive = Flatten(BuildTree(data));
            var iterative = FlattenIterative(BuildTree(data));
            Check(string.Join(",", RightChain(recursive)) == expected, "recursive chain");
            Check(string.Join(",", RightChain(iterative)) == expected, "iterative chain");
            Check(AllLeftNull(recursive) && AllLeftNull(iterative), "left pointers are empty");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
