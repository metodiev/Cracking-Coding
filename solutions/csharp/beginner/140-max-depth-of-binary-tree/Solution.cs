// Task 140 - Max depth of binary tree. (Tier: Beginner, CTCI Ch 4)
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

    public static int MaxDepth(TreeNode? node)
    {
        if (node == null) return 0;
        return 1 + Math.Max(MaxDepth(node.Left), MaxDepth(node.Right));
    }

    public static int MaxDepthIterative(TreeNode? root)
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
                if (node.Left != null) q.Enqueue(node.Left);
                if (node.Right != null) q.Enqueue(node.Right);
            }
        }
        return depth;
    }

    public static void Main()
    {
        Check(MaxDepth(BuildTree(1, 2, 3, 4, 5, null, 6)) == 3, "balanced three levels");
        Check(MaxDepth(null) == 0, "empty");
        Check(MaxDepth(BuildTree(7)) == 1, "single");
        Check(MaxDepth(BuildTree(1, 2, null, 3, null, 4)) == 4, "skewed");
        Check(MaxDepth(BuildTree(1, 2, 3, 4, 5, null, 6)) ==
              MaxDepthIterative(BuildTree(1, 2, 3, 4, 5, null, 6)), "recursive == iterative");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
