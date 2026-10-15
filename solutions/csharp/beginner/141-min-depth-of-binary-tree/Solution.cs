// Task 141 - Min depth of binary tree. (Tier: Beginner, CTCI Ch 4)
//
// Minimum depth = nodes on the shortest root-to-leaf path; a node with only
// one child is not a leaf, so [1, 2, null, 3] has min depth 3 (not 2).
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

    /// <summary>BFS level count: the first level holding a leaf is the shallowest one.</summary>
    public static int MinDepth(TreeNode? root)
    {
        if (root == null) return 0;
        var q = new Queue<TreeNode>();
        q.Enqueue(root);
        int depth = 0;
        while (q.Count > 0)
        {
            depth++;
            for (int i = q.Count; i > 0; i--)
            {
                var node = q.Dequeue();
                if (node.Left == null && node.Right == null) return depth;
                if (node.Left != null) q.Enqueue(node.Left);
                if (node.Right != null) q.Enqueue(node.Right);
            }
        }
        return depth;
    }

    /// <summary>Follow the only child instead of treating a missing child as an empty leaf.</summary>
    public static int MinDepthRecursive(TreeNode? node)
    {
        if (node == null) return 0;
        if (node.Left == null && node.Right == null) return 1;
        if (node.Left == null) return 1 + MinDepthRecursive(node.Right);
        if (node.Right == null) return 1 + MinDepthRecursive(node.Left);
        return 1 + Math.Min(MinDepthRecursive(node.Left), MinDepthRecursive(node.Right));
    }

    public static void Main()
    {
        Check(MinDepth(null) == 0 && MinDepthRecursive(null) == 0, "empty");
        Check(MinDepth(BuildTree(7)) == 1 && MinDepthRecursive(BuildTree(7)) == 1, "single node");
        Check(MinDepth(BuildTree(1, 2, null, 3)) == 3, "one-sided child is not a leaf");
        Check(MinDepthRecursive(BuildTree(1, 2, null, 3)) == 3, "one-sided child recursive");
        Check(MinDepth(BuildTree(3, 9, 20, null, null, 15, 7)) == 2, "sample tree");
        Check(MinDepth(BuildTree(1, 2, null, 3, null, 4)) == 4, "skewed chain");
        Check(MinDepth(BuildTree(1, 2, 3, 4, 5, null, 6)) ==
              MinDepthRecursive(BuildTree(1, 2, 3, 4, 5, null, 6)), "bfs == recursive");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
