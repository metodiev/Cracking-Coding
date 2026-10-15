// Task 151 - Path sum existence in tree. (Tier: Beginner, CTCI Ch 4)
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

    /// <summary>True iff a root-to-leaf path sums to target (leaf = both children null).</summary>
    public static bool HasPathSum(TreeNode? root, int target)
    {
        if (root == null) return false;
        if (root.Left == null && root.Right == null) return root.Val == target;
        return HasPathSum(root.Left, target - root.Val) || HasPathSum(root.Right, target - root.Val);
    }

    public static void Main()
    {
        var root = BuildTree(5, 4, 8, 11, null, 13, 4, 7, 2, null, null, null, 1);
        Check(HasPathSum(root, 22), "5+4+11+2 exists");
        Check(HasPathSum(root, 18), "5+8+4+1 exists");
        Check(!HasPathSum(root, 5), "5 is not a leaf so it is not a path end");
        Check(!HasPathSum(null, 0), "empty tree has no paths");
        Check(HasPathSum(BuildTree(1), 1), "single node match");
        Check(!HasPathSum(BuildTree(1), 2), "single node miss");
        Check(!HasPathSum(BuildTree(1, 2), 1), "one missing child is not a path end");
        Check(HasPathSum(BuildTree(1, 2), 3), "one-child path 1 -> 2");
        Check(HasPathSum(BuildTree(-2, null, -3), -5), "negative values");
        Check(HasPathSum(BuildTree(0, 1, 1), 1), "zero root");
        Check(!HasPathSum(BuildTree(0, 1, 1), 0), "zero root, no zero path");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
